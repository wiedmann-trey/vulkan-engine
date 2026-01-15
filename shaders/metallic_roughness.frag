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

float sampleShadowMap(sampler2D shadowMap, vec2 uv, vec2 offset, float compareDepth, float bias) {
    float pcfDepth = texture(shadowMap, uv + offset).r;
    return compareDepth + bias < pcfDepth ? 1.0 : 0.0;
}

float pcfShadow(sampler2D shadowMap, vec2 shadowCoord, float currentDepth, float bias) {
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    float shadow = 0.0;
    
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            shadow += sampleShadowMap(shadowMap, shadowCoord, 
                vec2(x, y) * texelSize, currentDepth, bias);
        }
    }
    
    return shadow / 9.0;
}

float sampleShadowCascade(vec3 worldPos, vec3 normal) {
    vec4 viewPos = sceneData.view * vec4(worldPos, 1.0);
    float depthValue = abs(viewPos.z);
    
    int cascadeIndex = 3;
    for (int i = 0; i < 3; i++) {
        if (depthValue < sceneData.cascadeSplits[i]) {
            cascadeIndex = i;
            break;
        }
    }

    vec4 shadowCoord = sceneData.shadowMatrices[cascadeIndex] * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;


    if (shadowCoord.x < 0.0 || shadowCoord.x > 1.0 || 
        shadowCoord.y < 0.0 || shadowCoord.y > 1.0
        ) {
        return 1.0;
    }

    float bias = max(0.005 * (1.0 - dot(normal, -normalize(sceneData.sunlightDirection.xyz))), 0.0005);
    float currentDepth = shadowCoord.z;
    bias = 0.0005f;
    
    float shadow;
    if (cascadeIndex == 0) {
        shadow = pcfShadow(shadowMap0, shadowCoord.xy, currentDepth, bias);
    } else if (cascadeIndex == 1) {
        shadow = pcfShadow(shadowMap1, shadowCoord.xy, currentDepth, bias);
    } else if (cascadeIndex == 2) {
        shadow = pcfShadow(shadowMap2, shadowCoord.xy, currentDepth, bias);
    } else {
        shadow = pcfShadow(shadowMap3, shadowCoord.xy, currentDepth, bias);
    }
    return 1.0 - shadow;
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
  vec3 color = sceneData.ambientColor.xyz * baseColor;
  vec3 lightDir = -normalize(sceneData.sunlightDirection.xyz);
  float shadowFactor = max(1-sceneData.sunlightDirection.w, sampleShadowCascade(inPosWorld, normal));
  float NoL = clamp(dot(normal, lightDir), 0.0, 1.0);
  color += BRDF(viewDir, lightDir, normal, roughness, f0, diffuseColor) * sceneData.sunlightColor.xyz * sceneData.sunlightColor.w * shadowFactor * NoL;

  outFragColor = vec4(color, alpha);
}
