#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

class DisneySubsurface : public BSDF {
public:
	DisneySubsurface(const Properties &props)
		: BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(0.1f)));
		m_roughness = props.getFloat("roughness", 0.0f);
		m_subsurface = props.getFloat("subsurface", 0.0f);
	}

	DisneySubsurface(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_base_color = static_cast<Texture *>(manager->getInstance(stream));
		m_roughness = stream->readFloat();
		m_subsurface = stream->readFloat();

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
					(F_SSwi * F_SSwo * (1.0f/(Frame::cosTheta(bRec.wi)+ Frame::cosTheta(bRec.wo)) - 0.5f) + 0.5f) * Frame::cosTheta(bRec.wo);
			}
		}

		return result_diffuse * (1.0f - m_subsurface) + result_subsurface * m_subsurface;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
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

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasSubsurface = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		if (!hasSubsurface)
			return Spectrum(0.0f);

		//sample diffuse 
		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.sampledComponent = 1;
		bRec.sampledType = EDiffuseReflection;

		bRec.eta = 1.0f;

		pdf = DisneySubsurface::pdf(bRec, ESolidAngle);

		/* unoptimized evaluation, explicit division of evaluation / pdf. */
		if (pdf == 0 || Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return DisneySubsurface::sample(bRec, pdf, sample);
	}


	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_base_color.get());
		stream->writeFloat(m_roughness);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "DisneySubsurface[" << endl
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

	//attributes
	ref<const Texture> m_base_color;
	Float m_roughness;
	Float m_subsurface;
};

// ================ Hardware shader implementation ================

class DisneySubsurfaceShader : public Shader {
public:
	DisneySubsurfaceShader(Renderer *renderer, const Texture *diffuseColor)
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

Shader *DisneySubsurface::createShader(Renderer *renderer) const {
	return new DisneySubsurfaceShader(renderer, m_base_color.get());
}

MTS_IMPLEMENT_CLASS(DisneySubsurfaceShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(DisneySubsurface, false, BSDF)
MTS_EXPORT_PLUGIN(DisneySubsurface, "Disney subsurface BRDF")
MTS_NAMESPACE_END
