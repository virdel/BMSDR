#version 330 core
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;

in vec4 Ambient;
in vec4 Diffuse;
in vec4 Specular;

out vec4 FragColor;

struct Material {
	bool useKd;
	bool useKs;
	bool useKn;
	sampler2D texture_diffuse1;
	sampler2D texture_specular1;
	sampler2D texture_normal1;
    float shininess;
}; 

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Light light;  
uniform Material material;
uniform vec3 viewPos;


void main(){

	//»·¾³¹â
	vec3 ambient,diffuse,specular;
	if(material.useKd){
		ambient=vec3(texture(material.texture_diffuse1,TexCoord))*light.ambient;
	}else{
		ambient=vec3(Ambient)*light.ambient;
	}

	//diffuse
	vec3 norm=normalize(Normal);
	vec3 lightdir=normalize(light.position-FragPos);
	float diff=max(dot(norm,lightdir),0.0);
	if(material.useKd){
		diffuse=diff*vec3(texture(material.texture_diffuse1,TexCoord))*light.diffuse;
	}else{
		diffuse=diff*vec3(Diffuse)*light.diffuse;
	}
	//specular
	vec3 viewDir=normalize(viewPos-FragPos);
	vec3 reflectDir=reflect(-lightdir,norm);
	float spec= pow(max(dot(viewDir,reflectDir),0.0),32);
	if(material.useKd){
		specular=(spec*vec3(texture(material.texture_specular1,TexCoord)))*light.specular;
	}else{
		specular=spec*vec3(Specular)*light.specular;
	}
	vec3 result=ambient+diffuse+specular;
	FragColor =vec4(result,1.0);

	//float gamma = 2.2;
    //FragColor.rgb = pow(FragColor.rgb, vec3(1.0/gamma));

	
}