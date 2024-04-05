#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

class DisneyMetal : public BSDF {
public:
	DisneyMetal(const Properties &props)
		: BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(0.1f)));
		m_roughness = props.getFloat("roughness", 0.0f);
		m_anisotropic = props.getFloat("anisotropic", 0.0f);
	}

	DisneyMetal(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_base_color = static_cast<Texture *>(manager->getInstance(stream));
		m_roughness = stream->readFloat();
		m_anisotropic = stream->readFloat();

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
				const Float alpha_min = 0.0001f;
				const Float roughness2 = m_roughness * m_roughness;
				const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
				const Float alpha_x = std::max(alpha_min, roughness2 / m_aspect);
				const Float alpha_y = std::max(alpha_min, roughness2 * m_aspect);

				const Float tmp = H.x * H.x / (alpha_x * alpha_x) + H.y * H.y / (alpha_y * alpha_y) + H.z * H.z;
				const Float Dm = INV_PI * 1.0f / (alpha_x * alpha_y * tmp * tmp);

				//Fresnel Term
				const Spectrum Fm = fresnel(m_base_color->eval(bRec.its), Hwi);

				//shadowing and masking
				const Float Gm = G_w(bRec.wi, alpha_x, alpha_y) * G_w(bRec.wo, alpha_x, alpha_y);

				//microfacet model
				result += Dm * Gm * Fm / (4.0f * abs(Frame::cosTheta(bRec.wi)));
			}
		}

		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		Float specProb = 0.0f;
		//diffuse pdf

		if (hasSpecular) {
			Vector H = bRec.wo + bRec.wi;   Float Hlen = H.length();
			if (Hlen == 0.0f) specProb = 0.0f;
			else
			{
				H = normalize(bRec.wo + bRec.wi);

				const Float Hwi = dot(bRec.wi, H);
				const Float Hwo = dot(bRec.wo, H);
				const Float alpha_min = 0.0001f;
				const Float roughness2 = m_roughness * m_roughness;
				const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
				const Float alpha_x = std::max(alpha_min, roughness2 / m_aspect);
				const Float alpha_y = std::max(alpha_min, roughness2 * m_aspect);

				const Float tmp = H.x * H.x / (alpha_x * alpha_x) + H.y * H.y / (alpha_y * alpha_y) + H.z * H.z;
				const Float Dm = INV_PI * 1.0f / (alpha_x * alpha_y * tmp * tmp);

				specProb = Dm + G_w(bRec.wi, alpha_x, alpha_y) * std::max(0.0f, Hwi) / Frame::cosTheta(bRec.wi);
			}
		}
		if (hasSpecular)
			return specProb;
		else
			return 0.0f;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);

		if (!hasSpecular)
			return Spectrum(0.0f);

		//sample specular

		const Float alpha_min = 0.0001f;
		const Float roughness2 = m_roughness * m_roughness;
		const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
		const Float alpha_x = std::max(alpha_min, roughness2 / m_aspect);
		const Float alpha_y = std::max(alpha_min, roughness2 * m_aspect);

		//hemisphere configuration
		Vector3f Vh(alpha_x * bRec.wi.x, alpha_y * bRec.wi.y, bRec.wi.z);
		Vh = normalize(Vh);

		//orthonormal basis
		Vector3f T0(0, 0, 1);
		Vector3f T1;
		if (Vh.z < 0.9999)
			T1 = DisneyMetal::cross(T0, Vh);
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
		Vector3f Ne = Vector3f(alpha_x * Nh.x, alpha_y * Nh.y, std::max(0.0f, Nh.z));
		Ne = normalize(Ne);

		bRec.wo = 2.0f * dot(bRec.wi, Ne) * Ne - bRec.wi;
		bRec.wo = normalize(bRec.wo);

		pdf = DisneyMetal::pdf(bRec, ESolidAngle);

		/* unoptimized evaluation, explicit division of evaluation / pdf. */
		if (pdf == 0 || Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return DisneyMetal::sample(bRec, pdf, sample);
	}


	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_base_color.get());
		stream->writeFloat(m_roughness);
		stream->writeFloat(m_anisotropic);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "DisneyMetal[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  base_color = " << indent(m_base_color->toString()) << ", " << endl
			<< "  roughness = " << m_roughness << ", " << endl
			<< "  roughness = " << m_anisotropic << ", " << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	//helper method
	inline Spectrum fresnel(const Spectrum& F0, const Float& c) const
	{
		return F0 + (Spectrum(0.1f) - F0) * pow(1.0 - c, 5.0f);
	}

	Float G_w(const Vector3f w, const Float m_alpha_x, const Float m_alpha_y) const
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
	Float m_roughness;
	Float m_anisotropic;
};

// ================ Hardware shader implementation ================

class DisneyMetalShader : public Shader {
public:
	DisneyMetalShader(Renderer *renderer, const Texture *diffuseColor)
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

Shader *DisneyMetal::createShader(Renderer *renderer) const {
	return new DisneyMetalShader(renderer, m_base_color.get());
}

MTS_IMPLEMENT_CLASS(DisneyMetalShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(DisneyMetal, false, BSDF)
MTS_EXPORT_PLUGIN(DisneyMetal, "Disney diffuse BRDF")
MTS_NAMESPACE_END
