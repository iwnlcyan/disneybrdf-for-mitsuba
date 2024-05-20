/*
	This file is part of Mitsuba, a physically based rendering system.

	Copyright (c) 2007-2014 by Wenzel Jakob and others.

	Mitsuba is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License Version 3
	as published by the Free Software Foundation.

	Mitsuba is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"

MTS_NAMESPACE_BEGIN

class DisneyGlass : public BSDF {
public:
	DisneyGlass(const Properties &props) : BSDF(props) {
		m_base_color = new ConstantSpectrumTexture(
			props.getSpectrum("base_color", Spectrum(0.1f)));
		m_roughness = props.getFloat("roughness", 0.0f);

		const Float alpha_min = 0.0001f;
		const Float roughness2 = m_roughness * m_roughness;
		const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
		m_alphaU = std::max(alpha_min, roughness2 / m_aspect);
		m_alphaV = std::max(alpha_min, roughness2 * m_aspect);

		m_eta = props.getFloat("eta", 0.0f);
		m_invEta = 1 / m_eta;

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();
	}

	DisneyGlass(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_roughness = stream->readFloat();
		m_anisotropic = stream->readFloat();
		m_eta = stream->readFloat();
		m_invEta = 1 / m_eta;

		const Float alpha_min = 0.0001f;
		const Float roughness2 = m_roughness * m_roughness;
		const Float m_aspect = sqrt(1.0f - 0.9f * m_anisotropic);
		m_alphaU = std::max(alpha_min, roughness2 / m_aspect);
		m_alphaV = std::max(alpha_min, roughness2 * m_aspect);

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t)m_type);
		stream->writeBool(m_sampleVisible);
		stream->writeFloat(m_eta);
		stream->writeFloat(m_roughness);
		stream->writeFloat(m_anisotropic);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		m_components.clear();

		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) == 0)
			return Spectrum(0.0f);

		/* Determine the type of interaction */
		bool reflect = Frame::cosTheta(bRec.wi)
			* Frame::cosTheta(bRec.wo) > 0;

		Vector H;
		if (reflect) {
			/* Stop if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 0)
				|| !(bRec.typeMask & EGlossyReflection))
				return Spectrum(0.0f);

			/* Calculate the reflection half-vector */
			H = normalize(bRec.wo + bRec.wi);
		}
		else {
			/* Stop if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 1)
				|| !(bRec.typeMask & EGlossyTransmission))
				return Spectrum(0.0f);

			/* Calculate the transmission half-vector */
			Float eta = Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;

			H = normalize(bRec.wi + bRec.wo*eta);
		}

		/* Ensure that the half-vector points into the
		   same hemisphere as the macrosurface normal */
		H *= math::signum(Frame::cosTheta(H));

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			/*m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),*/
			m_alphaU,
			m_alphaV,
			m_sampleVisible
		);

		/* Evaluate the microfacet normal distribution */
		const Float D = distr.eval(H);
		if (D == 0)
			return Spectrum(0.0f);

		/* Fresnel factor */
		const Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);

		/* Smith's shadow-masking function */
		const Float G = distr.G(bRec.wi, bRec.wo, H);

		if (reflect) {
			/* Calculate the total amount of reflection */
			Float value = F * D * G /
				(4.0f * std::abs(Frame::cosTheta(bRec.wi)));

			return m_base_color->eval(bRec.its) * value;
		}
		else {
			Float eta = Frame::cosTheta(bRec.wi) > 0.0f ? m_eta : m_invEta;

			/* Calculate the total amount of transmission */
			Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
			Float value = ((1 - F) * D * G * eta * eta
				* dot(bRec.wi, H) * dot(bRec.wo, H)) /
				(Frame::cosTheta(bRec.wi) * sqrtDenom * sqrtDenom);

			/* Missing term in the original paper: account for the solid angle
			   compression when tracing radiance -- this is necessary for
			   bidirectional methods */
			Float factor = (bRec.mode == ERadiance)
				? (Frame::cosTheta(bRec.wi) > 0 ? m_invEta : m_eta) : 1.0f;

			return colorsqrt(m_base_color->eval(bRec.its))
				* std::abs(value * factor * factor);
		}
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle)
			return 0.0f;

		/* Determine the type of interaction */
		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & EGlossyReflection)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
				&& (bRec.typeMask & EGlossyTransmission)),
			reflect = Frame::cosTheta(bRec.wi)
			* Frame::cosTheta(bRec.wo) > 0;

		Vector H;
		Float dwh_dwo;

		if (reflect) {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 0)
				|| !(bRec.typeMask & EGlossyReflection))
				return 0.0f;

			/* Calculate the reflection half-vector */
			H = normalize(bRec.wo + bRec.wi);

			/* Jacobian of the half-direction mapping */
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));
		}
		else {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 1)
				|| !(bRec.typeMask & EGlossyTransmission))
				return 0.0f;

			/* Calculate the transmission half-vector */
			Float eta = Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;

			H = normalize(bRec.wi + bRec.wo*eta);

			/* Jacobian of the half-direction mapping */
			Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
			dwh_dwo = (eta*eta * dot(bRec.wo, H)) / (sqrtDenom*sqrtDenom);
		}

		/* Ensure that the half-vector points into the
		   same hemisphere as the macrosurface normal */
		H *= math::signum(Frame::cosTheta(H));

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution sampleDistr(
			m_type,
			/*m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),*/
			m_alphaU,
			m_alphaV,
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Evaluate the microfacet model sampling density function */
		Float prob = sampleDistr.pdf(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, H);

		if (hasTransmission && hasReflection) {
			Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);
			prob *= reflect ? F : (1 - F);
		}

		return std::abs(prob * dwh_dwo);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & EGlossyReflection)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
				&& (bRec.typeMask & EGlossyTransmission)),
			sampleReflection = hasReflection;

		if (!hasReflection && !hasTransmission)
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			/*m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),*/
			m_alphaU,
			m_alphaV,
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		MicrofacetDistribution sampleDistr(distr);
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Sample M, the microfacet normal */
		Float microfacetPDF;
		const Normal m = sampleDistr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, sample, microfacetPDF);
		if (microfacetPDF == 0)
			return Spectrum(0.0f);

		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Spectrum weight(1.0f);

		if (hasReflection && hasTransmission) {
			if (bRec.sampler->next1D() > F)
				sampleReflection = false;
		}
		else {
			weight = Spectrum(hasReflection ? F : (1 - F));
		}

		if (sampleReflection) {
			/* Perfect specular reflection based on the microfacet normal */
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 0.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);

			weight *= m_base_color->eval(bRec.its);
		}
		else {
			if (cosThetaT == 0)
				return Spectrum(0.0f);

			/* Perfect specular transmission based on the microfacet normal */
			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
				return Spectrum(0.0f);

			/* Radiance must be scaled to account for the solid angle compression
			   that occurs when crossing the interface. */
			Float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			weight *= colorsqrt(m_base_color->eval(bRec.its)) * (factor * factor);
		}

		if (m_sampleVisible)
			weight *= distr.smithG1(bRec.wo, m);
		else
			weight *= std::abs(distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (microfacetPDF * Frame::cosTheta(bRec.wi)));

		return weight;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & EGlossyReflection)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
				&& (bRec.typeMask & EGlossyTransmission)),
			sampleReflection = hasReflection;

		if (!hasReflection && !hasTransmission)
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			/*m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),*/
			m_alphaU,
			m_alphaV,
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		MicrofacetDistribution sampleDistr(distr);
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Sample M, the microfacet normal */
		Float microfacetPDF;
		const Normal m = sampleDistr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, sample, microfacetPDF);
		if (microfacetPDF == 0)
			return Spectrum(0.0f);
		pdf = microfacetPDF;

		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Spectrum weight(1.0f);

		if (hasReflection && hasTransmission) {
			if (bRec.sampler->next1D() > F) {
				sampleReflection = false;
				pdf *= 1 - F;
			}
			else {
				pdf *= F;
			}
		}
		else {
			weight *= hasReflection ? F : (1 - F);
		}

		Float dwh_dwo;
		if (sampleReflection) {
			/* Perfect specular reflection based on the microfacet normal */
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);

			weight *= m_base_color->eval(bRec.its);

			/* Jacobian of the half-direction mapping */
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, m));
		}
		else {
			if (cosThetaT == 0)
				return Spectrum(0.0f);

			/* Perfect specular transmission based on the microfacet normal */
			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
				return Spectrum(0.0f);

			/* Radiance must be scaled to account for the solid angle compression
			   that occurs when crossing the interface. */
			Float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			weight *= colorsqrt(m_base_color->eval(bRec.its)) * (factor * factor);

			/* Jacobian of the half-direction mapping */
			Float sqrtDenom = dot(bRec.wi, m) + bRec.eta * dot(bRec.wo, m);
			dwh_dwo = (bRec.eta*bRec.eta * dot(bRec.wo, m)) / (sqrtDenom*sqrtDenom);
		}

		if (m_sampleVisible)
			weight *= distr.smithG1(bRec.wo, m);
		else
			weight *= std::abs(distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (microfacetPDF * Frame::cosTheta(bRec.wi)));

		pdf *= std::abs(dwh_dwo);

		return weight;
	}



	/*Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
	}*/

	std::string toString() const {
		std::ostringstream oss;
		oss << "DisneyGlass[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  eta = " << m_eta << "," << endl
			<< "  alphaU = " << m_alphaU << "," << endl
			<< "  alphaV = " << m_alphaV << "," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:

	inline Spectrum colorsqrt(const Spectrum& s) const
	{
		Spectrum result;
		for (size_t i = 0; i < SPECTRUM_SAMPLES; ++i) {
			result[i] = std::sqrt(s[i]);
		}
		return result;
	}

	MicrofacetDistribution::EType m_type;
	ref<const Texture> m_base_color;
	Float m_roughness;
	Float m_anisotropic;
	Float m_alphaU, m_alphaV;
	Float m_eta, m_invEta;
	bool m_sampleVisible;
};

/* Fake glass shader -- it is really hopeless to visualize
   this material in the VPL renderer, so let's try to do at least
   something that suggests the presence of a transparent boundary */
class DisneyGlassShader : public Shader {
public:
	DisneyGlassShader(Renderer *renderer, Float eta) :
		Shader(renderer, EBSDFShader) {
		m_flags = ETransparent;
	}

	Float getAlpha() const {
		return 0.3f;
	}

	void generateCode(std::ostringstream &oss,
		const std::string &evalName,
		const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "        return vec3(0.0);" << endl
			<< "    return vec3(inv_pi * cosTheta(wo));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}


	MTS_DECLARE_CLASS()
};

Shader *DisneyGlass::createShader(Renderer *renderer) const {
	return new DisneyGlassShader(renderer, m_eta);
}

MTS_IMPLEMENT_CLASS(DisneyGlassShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(DisneyGlass, false, BSDF)
MTS_EXPORT_PLUGIN(DisneyGlass, "Rough dielectric BSDF");
MTS_NAMESPACE_END
