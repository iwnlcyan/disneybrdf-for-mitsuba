#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

class DisneySheen : public BSDF {
public:
	DisneySheen(const Properties &props)
		: BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(0.1f)));
		m_sheen = props.getFloat("sheen", 0.0f);
		m_sheenTint = props.getFloat("sheen_tint", 0.0f);
		m_roughness = props.getFloat("roughness", 0.0f);
	}

	DisneySheen(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_base_color = static_cast<Texture *>(manager->getInstance(stream));
		m_sheen = stream->readFloat();
		m_sheenTint = stream -> readFloat();
		m_roughness = stream->readFloat();

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
				result += F_sheen * m_sheen + (m_base_color->eval(bRec.its) * INV_PI * Fwi * Fwo * Frame::cosTheta(bRec.wo));
			}
		}
		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
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

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		if (!hasDiffuse)
			return Spectrum(0.0f);

		//sample diffuse 
		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.sampledComponent = 1;
		bRec.sampledType = EDiffuseReflection;

		bRec.eta = 1.0f;

		pdf = DisneySheen::pdf(bRec, ESolidAngle);

		/* unoptimized evaluation, explicit division of evaluation / pdf. */
		if (pdf == 0 || Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return DisneySheen::sample(bRec, pdf, sample);
	}


	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_base_color.get());
		stream->writeFloat(m_sheen);
		stream->writeFloat(m_sheenTint);
		stream->writeFloat(m_roughness);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "DisneySheen[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  base_color = " << indent(m_base_color->toString()) << ", " << endl
			<< "  sheen_tint = " << m_sheenTint << ", " << endl
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

	//attributes
	ref<const Texture> m_base_color;
	Float m_sheenTint;
	Float m_roughness;
	Float m_sheen;
};

// ================ Hardware shader implementation ================

class DisneySheenShader : public Shader {
public:
	DisneySheenShader(Renderer *renderer, const Texture *diffuseColor)
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

Shader *DisneySheen::createShader(Renderer *renderer) const {
	return new DisneySheenShader(renderer, m_base_color.get());
}

MTS_IMPLEMENT_CLASS(DisneySheenShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(DisneySheen, false, BSDF)
MTS_EXPORT_PLUGIN(DisneySheen, "Disney diffuse BRDF")
MTS_NAMESPACE_END
