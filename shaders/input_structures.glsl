layout(set = 0, binding = 0) uniform  SceneData{   

	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; // w = casts shadows 
	vec4 sunlightColor; // w = sun strength
  mat4 shadowMatrices[4];
  vec4 cascadeSplits;
} sceneData;

layout(set = 0, binding = 1) uniform sampler2D shadowMap0;
layout(set = 0, binding = 2) uniform sampler2D shadowMap1;
layout(set = 0, binding = 3) uniform sampler2D shadowMap2;
layout(set = 0, binding = 4) uniform sampler2D shadowMap3;

layout(set = 1, binding = 0) uniform GLTFMaterialData{   

	vec4 colorFactors;
	vec4 metal_rough_factors;
	
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;

const float PI = 3.14159265359;
