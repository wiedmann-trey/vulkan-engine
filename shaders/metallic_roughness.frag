#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inPosWorld;

layout (location = 0) out vec4 outFragColor;

// brdf model from https://google.github.io/filament/Filament.md.html#materialsystem/standardmodelsummary
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * f * f);
}

vec3 F_Schlick(float u, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - u, 5.0);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
    float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    return 0.5 / (GGXV + GGXL);
}

float Fd_Lambert() {
    return 1.0 / PI;
}

vec3 BRDF(vec3 v, vec3 l, vec3 n, float roughness, vec3 f0, vec3 diffuseColor) {
    vec3 h = normalize(v + l);

    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = clamp(dot(n, l), 0.0, 1.0);
    float NoH = clamp(dot(n, h), 0.0, 1.0);
    float LoH = clamp(dot(l, h), 0.0, 1.0);

    float D = D_GGX(NoH, roughness);
    vec3  F = F_Schlick(LoH, f0);
    float V = V_SmithGGXCorrelated(NoV, NoL, roughness);

    // specular BRDF
    vec3 Fr = (D * V) * F;

    // diffuse BRDF
    vec3 Fd = diffuseColor * Fd_Lambert();

    return Fr + Fd;
}

void main() 
{
  vec3 camPos = inverse(sceneData.view)[3].xyz;
  vec3 viewDir = normalize(camPos - inPosWorld);
  vec3 normal = normalize(inNormal);

  float metallic = materialData.metal_rough_factors.x * texture(metalRoughTex, inUV).z;
  float roughness = materialData.metal_rough_factors.y * texture(metalRoughTex, inUV).y;
	vec3 baseColor = inColor * texture(colorTex,inUV).xyz * materialData.colorFactors.xyz;
  float alpha = texture(colorTex, inUV).w * materialData.colorFactors.w;

  vec3 diffuseColor = (1.0 - metallic) * baseColor;
  vec3 f0 = mix(vec3(0.04), baseColor, metallic);
  roughness = clamp(roughness * roughness,  0.089f, 1.f);

  // eval brdf for all lights
  vec3 color = vec3(0.f);
  vec3 lightDir = -normalize(sceneData.sunlightDirection.xyz);
  float NoL = clamp(dot(normal, lightDir), 0.0, 1.0);
  color += BRDF(viewDir, lightDir, normal, roughness, f0, diffuseColor) * sceneData.sunlightColor.xyz * NoL;

	outFragColor = vec4(color, alpha);
}
