#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#include <mitsuba/render/sampler.h>
#include "microfacet.h"
#include "ior.h"

MTS_NAMESPACE_BEGIN

class Disney : public BSDF {
public:
	Disney(const Properties &props)
		: BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(1.0f)));
		m_subsurface = props.getFloat("subsurface", 0.0f);
		m_sheen = props.getFloat("sheen", 0.0f);
		m_sheenTint = props.getFloat("sheen_tint", 0.0f);
		m_clearcoat = props.getFloat("clearcoat", 0.0f);
		m_clearcoatGloss = props.getFloat("clearcoat_gloss", 0.0f);
		m_specular = props.getFloat("specular", 0.0f);
		m_specTint = props.getFloat("specTint", 0.0f);
		m_specTrans = props.getFloat("specTrans", 0.0f);
		m_roughness = props.getFloat("roughness", 0.0f);
		m_anisotropic = props.getFloat("anisotropic", 0.0f);
		m_metallic = props.getFloat("metallic", 0.0f);
		m_eta = props.getFloat("eta", 0.0f);

		const Float alpha_min = 0.0001f;
		const Float roughness2 = m_roughness * m_roughness;
		const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
		m_alphaU = std::max(alpha_min, roughness2 / m_aspect);
		m_alphaV = std::max(alpha_min, roughness2 * m_aspect);
		m_invEta = 1 / m_eta;

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();
	}

	Disney(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_base_color = static_cast<Texture *>(manager->getInstance(stream));
		m_subsurface = stream->readFloat();
		m_sheen = stream->readFloat();
		m_sheenTint = stream->readFloat();
		m_clearcoat = stream->readFloat();
		m_clearcoatGloss = stream->readFloat();
		m_specular = stream->readFloat();
		m_specTint = stream->readFloat();
		m_specTrans = stream->readFloat();
		m_roughness = stream->readFloat();
		m_anisotropic = stream->readFloat();
		m_metallic = stream->readFloat();
		m_eta = stream->readFloat();
		m_invEta = 1 / m_eta;

		const Float alpha_min = 0.0001f;
		const Float roughness2 = m_roughness * m_roughness;
		const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
		m_alphaU = std::max(alpha_min, roughness2 / m_aspect);
		m_alphaV = std::max(alpha_min, roughness2 * m_aspect);

		configure();
	}

	void configure() {
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide);
		m_components.push_back(EDiffuseReflection | EFrontSide);
		m_usesRayDifferentials = false;

		BSDF::configure();
	}

	Spectrum evalDiffuse(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);
		bool hasSubsurface = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		if ((!hasDiffuse && !hasSubsurface)
			|| Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0) {
			//std::cout << "no diffuse" << endl;
			return Spectrum(0.0f);
		}
		/* eval diffusse */
		Spectrum result(0.0f);
		Spectrum result_diffuse(0.0f);
		Spectrum result_subsurface(0.0f);

		if (hasDiffuse) {
			Vector H = normalize(bRec.wo + bRec.wi);
			if (Frame::cosTheta(H) > 0.0f)
			{
				//half vector
				const Vector Phi = bRec.wo + bRec.wi;
				const Vector H = normalize(Phi);
				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				//Fresnel Term
				const Float F_D90 = 0.5f + 2.0f * m_roughness * Hwo * Hwo;
				const Float Fwi = fresnel(F_D90, Frame::cosTheta(bRec.wi));
				const Float Fwo = fresnel(F_D90, Frame::cosTheta(bRec.wo));
				//Diffuse
				result_diffuse += m_base_color->eval(bRec.its) * INV_PI * Fwi * Fwo * Frame::cosTheta(bRec.wo);
			}
		}

		if (hasSubsurface) {
			Vector H = normalize(bRec.wo + bRec.wi);
			if (Frame::cosTheta(H) > 0.0f)
			{
				//half vector
				const Vector Phi = bRec.wo + bRec.wi;
				const Vector H = normalize(Phi);
				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				//Fresnel Term
				const Float F_SS90 = m_roughness * Hwo * Hwo;
				const Float F_SSwi = fresnelSS(F_SS90, Frame::cosTheta(bRec.wi));
				const Float F_SSwo = fresnelSS(F_SS90, Frame::cosTheta(bRec.wo));
				//Diffuse
				result_subsurface += 1.25f * m_base_color->eval(bRec.its) * INV_PI *
					(F_SSwi * F_SSwo * (1.0f / (Frame::cosTheta(bRec.wi) + Frame::cosTheta(bRec.wo)) - 0.5f) + 0.5f) * Frame::cosTheta(bRec.wo);
			}
		}

		return result_diffuse * (1.0f - m_subsurface) + result_subsurface * m_subsurface;
	}

	Float pdfDiffuse(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSubsurface = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Float subsurfaceProb = 0.0f;
		//diffuse pdf
		if (hasSubsurface)
			subsurfaceProb = warp::squareToCosineHemispherePdf(bRec.wo);
		if (hasSubsurface)
			return subsurfaceProb;
		//subsurface pdf
		else
			return 0.0f;
	}

	void sampleDiffuse(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		//sample diffuse 
		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.sampledComponent = 1;
		bRec.sampledType = EDiffuseReflection;

		bRec.eta = 1.0f;

		pdf = Disney::pdfDiffuse(bRec, ESolidAngle);

	}

	void sampleDiffuse(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		Disney::sampleDiffuse(bRec, pdf, sample);
	}

	Spectrum evalMetal(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		if ((!hasSpecular)
			|| Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0) {
			//std::cout << "no diffuse" << endl;
			return Spectrum(0.0f);
		}
		/* eval diffusse */
		Spectrum result(0.0f);
		//Spectrum result_diffuse(0.0f);
		//Spectrum result_anisotropic(0.0f);

		if (hasSpecular) {
			Vector H = normalize(bRec.wo + bRec.wi);
			if (Frame::cosTheta(H) > 0.0f)
			{
				//NDF
				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);

				const Float tmp = H.x * H.x / (m_alphaU * m_alphaU) + H.y * H.y / (m_alphaV * m_alphaV) + H.z * H.z;
				const Float Dm = INV_PI * 1.0f / (m_alphaU * m_alphaV * tmp * tmp);

				//Fresnel Term
				const Spectrum Fm_hat = fresnelMetal(bRec, Hwo);

				//shadowing and masking
				const Float Gm = G_w(bRec.wi, m_alphaU, m_alphaV) * G_w(bRec.wo, m_alphaU, m_alphaV);

				//microfacet model
				result += Dm * Gm * Fm_hat / (4.0f * abs(Frame::cosTheta(bRec.wi)));
			}
		}

		return result;
	}

	Float pdfMetal(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		Float specProb = 0.0f;

		if (hasSpecular) {
			Vector H = bRec.wo + bRec.wi;   Float Hlen = H.length();
			if (Hlen == 0.0f) specProb = 0.0f;
			else
			{
				H = normalize(bRec.wo + bRec.wi);

				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);

				const Float tmp = H.x * H.x / (m_alphaU * m_alphaU) + H.y * H.y / (m_alphaV * m_alphaV) + H.z * H.z;
				const Float Dm = INV_PI * 1.0f / (m_alphaU * m_alphaV * tmp * tmp);

				specProb = Dm + G_w(bRec.wi, m_alphaU, m_alphaV) * std::max(0.0f, Hwi) / Frame::cosTheta(bRec.wi);
			}
		}
		if (hasSpecular)
			return specProb;
		else
			return 0.0f;
	}

	void sampleMetal(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		//hemisphere configuration
		Vector3f Vh(m_alphaU * bRec.wi.x, m_alphaV * bRec.wi.y, bRec.wi.z);
		Vh = normalize(Vh);

		//orthonormal basis
		Vector3f T0(0, 0, 1);
		Vector3f T1;
		if (Vh.z < 0.9999)
			T1 = Disney::cross(T0, Vh);
		else
			T1 = Vector3f(1, 0, 0);
		T1 = normalize(T1);
		Vector3f T2 = cross(Vh, T1);

		// reprojection to hemisphere
		Float r = sqrt(sample.x);
		Float phi = 2.0f * M_PI * sample.y;
		Float t1 = r * cos(phi);
		Float t2 = r * sin(phi);
		Float s = 0.5f * (1.0f + Vh.z);
		t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

		Vector3f Nh = t1 * T1 + t2 * T2 + sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

		//ellipsoid configuration
		Vector3f Ne = Vector3f(m_alphaU * Nh.x, m_alphaV * Nh.y, std::max(0.0f, Nh.z));
		Ne = normalize(Ne);

		bRec.wo = 2.0f * dot(bRec.wi, Ne) * Ne - bRec.wi;
		bRec.wo = normalize(bRec.wo);

		pdf = Disney::pdfMetal(bRec, ESolidAngle);
	}

	void sampleMetal(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		Disney::sampleMetal(bRec, pdf, sample);
	}

	Spectrum evalClearcoat(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		if ((!hasSpecular)
			|| Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0) {
			//std::cout << "no diffuse" << endl;
			return Spectrum(0.0f);
		}

		Spectrum result(0.0f);
		if (hasSpecular) {
			Vector H = normalize(bRec.wo + bRec.wi);
			if (Frame::cosTheta(H) > 0.0f)
			{
				//NDF
				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				const Float alpha_g = (1 - m_clearcoatGloss) * 0.1f + m_clearcoatGloss * 0.001;

				const Float tmp = M_PI * log(alpha_g * alpha_g) + (1.0f + (alpha_g * alpha_g - 1.0f) * H.z * H.z);
				const Float Dc = (alpha_g * alpha_g - 1.0f) / tmp;

				//Fresnel Term
				const Spectrum Fc = fresnelClearcoat(Hwi);

				//shadowing and masking
				const Float Gc = Gc_w(bRec.wi, 0.25f, 0.25f) * Gc_w(bRec.wo, 0.25f, 0.25f);

				//microfacet model
				result += Dc * Gc * Fc / (4.0f * abs(Frame::cosTheta(bRec.wi)));
			}
		}

		return result;
	}

	Float pdfClearcoat(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		Float specProb = 0.0f;

		if (hasSpecular) {
			Vector H = bRec.wo + bRec.wi;   Float Hlen = H.length();
			if (Hlen == 0.0f) specProb = 0.0f;
			else
			{
				H = normalize(bRec.wo + bRec.wi);

				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				const Float alpha_g = (1.0f - m_clearcoatGloss) * 0.1f + m_clearcoatGloss * 0.001f;

				const Float tmp = M_PI * log(alpha_g * alpha_g) + (1.0f + (alpha_g * alpha_g - 1.0f) * H.z * H.z);
				const Float Dc = (alpha_g * alpha_g - 1.0f) / tmp;

				specProb = Dc * abs(Frame::cosTheta(H)) / (4.0f * abs(Hwo));
			}
		}
		if (hasSpecular)
			return specProb;
		else
			return 0.0f;
	}

	void sampleClearcoat(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);


		/* Side check */
		if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(bRec.wi) <= 0)
			return;

		//sample specular
		const Float alpha_g = (1.0f - m_clearcoatGloss) * 0.1f + m_clearcoatGloss * 0.001;

		float h_elevation = sqrt((1.0f - pow(alpha_g * alpha_g, 2.0f), 1.0f - sample.x) / (1.0f - alpha_g * alpha_g));
		h_elevation = acos(math::clamp(h_elevation, -1.f, 1.f));
		float h_azimuth = 2.0f * M_PI * sample.y;
		Vector3f wh;
		wh.x = sin(h_elevation) * cos(h_azimuth);
		wh.y = sin(h_elevation) * sin(h_azimuth);
		wh.z = cos(h_elevation);
		wh = normalize(wh);
		const Vector3f N = BSDF::getFrame(bRec.its).toWorld(wh);

		bRec.wo = 2.0f * dot(bRec.wi, N) * N - bRec.wi;

		pdf = Disney::pdfClearcoat(bRec, ESolidAngle);
	}

	void sampleClearcoat(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		Disney::sampleClearcoat(bRec, pdf, sample);
	}

	Spectrum evalSheen(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		if ((!hasDiffuse)
			|| Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0) {
			//std::cout << "no diffuse" << endl;
			return Spectrum(0.0f);
		}
		/* eval diffusse */
		Spectrum result(0.0f);
		if (hasDiffuse) {
			Vector H = normalize(bRec.wo + bRec.wi);
			if (Frame::cosTheta(H) > 0.0f)
			{
				Spectrum C_tint = (m_base_color->eval(bRec.its).getLuminance() > 0) ?
					(m_base_color->eval(bRec.its) / m_base_color->eval(bRec.its).getLuminance()) : Spectrum(1.0f);
				Spectrum C_sheen = (1.0f - m_sheenTint) * Spectrum(1.0f) + m_sheenTint * C_tint;
				Spectrum F_sheen = C_sheen * pow(1 - abs(dot(bRec.wo, H)), 5) * abs(Frame::cosTheta(bRec.wo));

				//half vector
				const Vector Phi = bRec.wo + bRec.wi;
				const Vector H = normalize(Phi);
				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				//Fresnel Term
				const Float F_D90 = 0.5f + 2.0f * m_roughness * Hwo * Hwo;
				const Float Fwi = fresnel(F_D90, Frame::cosTheta(bRec.wi));
				const Float Fwo = fresnel(F_D90, Frame::cosTheta(bRec.wo));
				//Diffuse
				result += F_sheen;
			}
		}
		return result;
	}

	Float pdfSheen(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Float diffuseProb = 0.0f;
		//diffuse pdf
		if (hasDiffuse)
			diffuseProb = warp::squareToCosineHemispherePdf(bRec.wo);
		if (hasDiffuse)
			return diffuseProb;
		//subsurface pdf
		else
			return 0.0f;
	}











	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return Spectrum(0.0f);

		Spectrum principledBSDF(0.0f);

		principledBSDF = (1.0f - m_metallic) * evalDiffuse(bRec, ESolidAngle)
			+ (1.0f - m_metallic) * m_sheen * evalSheen(bRec, ESolidAngle)
			+ evalMetal(bRec, ESolidAngle)
			+ 0.25 * m_clearcoat * evalClearcoat(bRec, ESolidAngle);

		return principledBSDF;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		Float principledPDF;
		Vector H = normalize(bRec.wo + bRec.wi);
		const Float Hwi = dot(bRec.wi, H);
		const Float Hwo = dot(bRec.wo, H);

		Float diffuseWeight = 1.0f - m_metallic;
		Float metalWeight = 1.0f;
		Float clearcoatWeight = 0.25 * m_clearcoat;

		Vector3f pdf(diffuseWeight, metalWeight, clearcoatWeight);
		pdf = normalize(pdf);

		principledPDF = pdf.x * pdfDiffuse(bRec, ESolidAngle)
			+ pdf.y * pdfMetal(bRec, ESolidAngle) / (4.0f * abs(Hwo))
			+ pdf.z * pdfClearcoat(bRec, ESolidAngle) / (4.0f * abs(Hwo));
			
		return principledPDF;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		if (Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		Float diffuseWeight = 1.f - m_metallic;
		Float metalWeight = 1.f;
		Float clearcoatWeight = 0.25 * m_clearcoat;

		// Construct cdf
		Vector3f cdf(diffuseWeight, metalWeight, clearcoatWeight);
		cdf /= cdf.x + cdf.y + cdf.z;
		cdf.y = cdf.x + cdf.y;
		cdf.z = cdf.y + cdf.z;

		if (sample.x < cdf.x) // Diffuse
		{
			sample.x /= cdf.x;
			sampleDiffuse(bRec, sample);
		}
		else if (sample.x < cdf.y) // Metal
		{
			sample.x = (sample.x - cdf.x) / (cdf.y - cdf.x);
			sampleMetal(bRec, sample);
		}
		else // Clearcoat
		{
			sample.x = (sample.x - cdf.y) / (1 - cdf.y);
			sampleClearcoat(bRec, sample);
		}

		Float pdf_disney = Disney::pdf(bRec, ESolidAngle);

		if (pdf_disney > 0)
			return eval(bRec, ESolidAngle) / pdf_disney;//* std::max(Frame::cosTheta(bRec.wo), 0.0f)
		else
			return Spectrum(0.0f);

	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return Disney::sample(bRec, pdf, sample);
	}


	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_base_color.get());
		stream->writeFloat(m_subsurface);
		stream->writeFloat(m_sheen);
		stream->writeFloat(m_sheenTint);
		stream->writeFloat(m_clearcoat);
		stream->writeFloat(m_clearcoatGloss);
		stream->writeFloat(m_specular);
		stream->writeFloat(m_specTint);
		stream->writeFloat(m_specTrans);
		stream->writeFloat(m_roughness);
		stream->writeFloat(m_anisotropic);
		stream->writeFloat(m_metallic);
		stream->writeFloat(m_eta);

		stream->writeUInt((uint32_t)m_type);
		stream->writeBool(m_sampleVisible);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Disney[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  base_color = " << indent(m_base_color->toString()) << ", " << endl
			<< "  roughness = " << m_roughness << ", " << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	//helper method
	inline Float fresnel(const Float& F_D90, const Float& c) const
	{
		return 1.0f + (F_D90 - 1.0f)*pow(1.0 - c, 5.0f);
	}
	inline Float fresnelSS(const Float& F_SS90, const Float& c) const
	{
		return 1.0f + (F_SS90 - 1.0f)*pow(1.0 - c, 5.0f);
	}

	inline Spectrum fresnelMetal(const BSDFSamplingRecord &bRec, const Float& c) const
	{
		return C0(bRec) + (Spectrum(1.0f) - C0(bRec)) * pow(1.0 - c, 5.0f);
	}

	inline Spectrum C0(const BSDFSamplingRecord &bRec) const
	{
		return m_specular * R_0(m_eta) * (1.f - m_metallic) * K_s(bRec) + m_metallic * m_base_color->eval(bRec.its);
	}

	inline Spectrum C_tint(const BSDFSamplingRecord &bRec) const
	{
		return (m_base_color->eval(bRec.its).getLuminance() > 0) ?
			(m_base_color->eval(bRec.its) / m_base_color->eval(bRec.its).getLuminance()) : Spectrum(1.0f);
	}

	inline Spectrum K_s(const BSDFSamplingRecord &bRec) const
	{
		return (1.0f - m_specTint) * Spectrum(1.0f) + m_specTint * C_tint(bRec);
	}

	Float G_w(const Vector3f w, const Float m_alpha_x, const Float m_alpha_y) const
	{
		const Float Lambda_w = (sqrt(1.0f + (pow(w.x * m_alpha_x, 2) + pow(w.y * m_alpha_y, 2)) / pow(w.z, 2)) - 1.0f) / 2.0f;
		return 1.0f / (1.0f + Lambda_w);
	}

	Spectrum R_0(const Float eta) const
	{
		return Spectrum(1.0f) * (eta - 1.0f) * (eta - 1.0f) / (eta + 1.0f) / (eta + 1.0f);
	}

	Spectrum fresnelClearcoat(const Float& c) const
	{
		return R_0(1.5f) + (Spectrum(1.0f) - R_0(1.5f)) * pow(1.0 - c, 5.0f);
	}

	Float Gc_w(const Vector3f w, const Float m_alpha_x, const Float m_alpha_y) const
	{
		const Float Lambda_w = (sqrt(1.0f + (pow(w.x * m_alpha_x, 2) + pow(w.y * m_alpha_y, 2)) / pow(w.z, 2)) - 1.0f) / 2.0f;
		return 1.0f / (1.0f + Lambda_w);
	}

	Vector3f cross(const Vector3f &a, const Vector3f &b) const
	{
		Vector3f c(0.0f);
		c.x = a.y*b.z - a.z*b.y;
		c.y = a.z*b.x - a.x*b.z;
		c.z = a.x*b.y - a.y*b.x;
		return c;
	}

	//attributes
	ref<const Texture> m_base_color;
	Float m_subsurface;
	Float m_sheenTint;
	Float m_sheen;
	Float m_clearcoat;
	Float m_clearcoatGloss;
	Float m_specular;
	Float m_specTint;
	Float m_specTrans;
	Float m_roughness;
	Float m_anisotropic;
	Float m_metallic;

	MicrofacetDistribution::EType m_type;
	Float m_alphaU, m_alphaV;
	Float m_eta, m_invEta;
	bool m_sampleVisible;
};

// ================ Hardware shader implementation ================

class DisneyShader : public Shader {
public:
	DisneyShader(Renderer *renderer, const Texture *diffuseColor)
		: Shader(renderer, EBSDFShader), m_base_color(diffuseColor) {
		m_base_colorShader = renderer->registerShaderForResource(m_base_color.get());
		m_flags = ETransparent;
	}

	bool isComplete() const {
		return m_base_color.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_base_color.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_base_colorShader.get());
	}

	void generateCode(std::ostringstream &oss,
		const std::string &evalName,
		const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "        return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_base_color;
	ref<Shader> m_base_colorShader;
};

Shader *Disney::createShader(Renderer *renderer) const {
	return new DisneyShader(renderer, m_base_color.get());
}

MTS_IMPLEMENT_CLASS(DisneyShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(Disney, false, BSDF)
MTS_EXPORT_PLUGIN(Disney, "Disney BRDF")
MTS_NAMESPACE_END