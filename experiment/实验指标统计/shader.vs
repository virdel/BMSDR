#version 330 core
layout (location =0) in vec3 aPos;
layout (location=1 ) in vec3 aNor;
layout (location=2 ) in vec2 Tex;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;
out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec4 Ambient;
out vec4 Diffuse;
out vec4 Specular;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform Mat{
	vec4 aAmbient;
	vec4 aDiffuse;
	vec4 aSpecular;
};
void main(){
	gl_Position =projection* view*model*vec4(aPos.x, aPos.y, aPos.z, 1.0);
	FragPos=vec3(model*vec4(aPos,1.0));
	Normal=vec3(model*vec4(aNor,1.0));
	TexCoord=Tex;
	Ambient=aAmbient;
	Diffuse=aDiffuse;
	Specular=aSpecular;

	vec3 T=normalize(vec3(model*vec4(tangent,0.0)));
	vec3 B=normalize(vec3(model*vec4(bitangent,0.0)));
	vec3 N=normalize(vec3(model*vec4(aNor,0.0)));


}
