#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

class DisneyClearcoat : public BSDF {
public:
	DisneyClearcoat(const Properties &props)
		: BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(0.1f)));
		m_roughness = props.getFloat("roughness", 0.0f);
		m_clearcoat = props.getFloat("clearcoat", 0.0f);
		m_clearcoatGloss = props.getFloat("clearcoat_gloss", 0.0f);
	}

	DisneyClearcoat(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_roughness = stream->readFloat();
		m_clearcoatGloss = stream->readFloat();
		m_clearcoat = stream->readFloat();

		configure();
	}

	void configure() {
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide);
		m_components.push_back(EDiffuseReflection | EFrontSide);
		m_usesRayDifferentials = false;

		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		/* sanity check */
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		/* which components to eval */
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		/* eval spec */
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
				const Spectrum Fc = fresnel(Hwi);

				//shadowing and masking
				const Float Gc = Gc_w(bRec.wi, 0.25f, 0.25f) * Gc_w(bRec.wo, 0.25f, 0.25f);

				//microfacet model
				result += Dc * Gc * Fc / (4.0f * abs(Frame::cosTheta(bRec.wi)));
			}
		}

		/* eval diffuse */
		if (hasDiffuse) {
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
					const Float Fwi = fresnelD(F_D90, Frame::cosTheta(bRec.wi));
					const Float Fwo = fresnelD(F_D90, Frame::cosTheta(bRec.wo));
					//Diffuse
					result += m_base_color->eval(bRec.its) * INV_PI * Fwi * Fwo * Frame::cosTheta(bRec.wo);
				}
			}
		}
		// Done.
		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Float diffuseProb = 0.0f, specProb = 0.0f;

		//* diffuse pdf */
		if (hasDiffuse)
			diffuseProb = warp::squareToCosineHemispherePdf(bRec.wo);

		/* specular pdf */
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

		Float m_specularSamplingWeight = 0.75f * m_clearcoat;

		if (hasDiffuse && hasSpecular)
			return m_specularSamplingWeight * specProb + (1.0f - m_specularSamplingWeight) * diffuseProb;
		else if (hasDiffuse)
			return diffuseProb;
		else if (hasSpecular)
			return specProb;
		else
			return 0.0f;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		if (!hasSpecular && !hasDiffuse)
			return Spectrum(0.0f);

		Float m_specularSamplingWeight = 0.75f * m_clearcoat;

		// determine which component to sample
		bool choseSpecular = hasSpecular;
		if (hasDiffuse && hasSpecular) {
			if (sample.x <= m_specularSamplingWeight) {
				sample.x /= m_specularSamplingWeight;
			}
			else {
				sample.x = (sample.x - m_specularSamplingWeight)
					/ (1.0f - m_specularSamplingWeight);
				choseSpecular = false;
			}
		}

		/* sample specular */
		if (choseSpecular) {
			//sample specular
			const Float alpha_g = (1.0f - m_clearcoatGloss) * 0.1f + m_clearcoatGloss * 0.001;

			//hemisphere configuration
			Vector3f Vh(alpha_g * bRec.wi.x, alpha_g * bRec.wi.y, bRec.wi.z);
			Vh = normalize(Vh);

			//orthonormal basis
			Vector3f T0(0, 0, 1);
			Vector3f T1;
			if (Vh.z < 0.9999)
				T1 = DisneyClearcoat::cross(T0, Vh);
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
			Vector3f Ne = Vector3f(alpha_g * Nh.x, alpha_g * Nh.y, std::max(0.0f, Nh.z));
			Ne = normalize(Ne);

			bRec.wo = 2.0f * dot(bRec.wi, Ne) * Ne - bRec.wi;
			bRec.wo = normalize(bRec.wo);

			/* sample diffuse */
		}
		else {
			bRec.wo = warp::squareToCosineHemisphere(sample);
			bRec.sampledComponent = 1;
			bRec.sampledType = EDiffuseReflection;
		}
		bRec.eta = 1.0f;
		
		pdf = DisneyClearcoat::pdf(bRec, ESolidAngle);

		/* unoptimized evaluation, explicit division of evaluation / pdf. */
		if (pdf == 0 || Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return DisneyClearcoat::sample(bRec, pdf, sample);
	}

	Frame getFrame(const Intersection &its) const {
		Frame result;
		Normal n;

		Frame frame = BSDF::getFrame(its);
		result.n = normalize(frame.toWorld(n));

		result.s = normalize(its.dpdu - result.n
			* dot(result.n, its.dpdu));

		result.t = cross(result.n, result.s);

		return result;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
		manager->serialize(stream, m_base_color.get());
		stream->writeFloat(m_roughness);
		stream->writeFloat(m_clearcoat);
		stream->writeFloat(m_clearcoatGloss);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "DisneyClearcoat[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  roughness = " << m_clearcoatGloss << ", " << endl
			<< "]";
		return oss.str();
	}


	MTS_DECLARE_CLASS()
private:
	//helper method
	//helper method
	inline Float fresnelD(const Float& F_D90, const Float& c) const
	{
		return 1.0f + (F_D90 - 1.0f)*pow(1.0 - c, 5.0f);
	}

	Spectrum R_0(const Float eta) const
	{
		return Spectrum(1.0f) * (eta - 1.0f) * (eta - 1.0f) / (eta + 1.0f) / (eta + 1.0f);
	}

	Spectrum fresnel(const Float& c) const
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

	ref<const Texture> m_base_color;
	Float m_roughness;
	Float m_clearcoatGloss;
	Float m_clearcoat;
};


MTS_IMPLEMENT_CLASS_S(DisneyClearcoat, false, BSDF)
MTS_EXPORT_PLUGIN(DisneyClearcoat, "Disney clearcoat BRDF")
MTS_NAMESPACE_END
