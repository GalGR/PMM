#include "Vector4.h"

Vector4::Vector4(const Vector3 &vec3) : Vector4(vec3.x, vec3.y, vec3.z) {}
Vector4::operator Vector3() const { return Vector4::euclid(*this); }
Vector3 Vector4::euclid(const Vector4 &vector) {
	Vector3 vector_euclid;
	for (int i = 0; i < 3; ++i) {
		vector_euclid[i] = vector(i);
	}
	return vector_euclid;
}
Vector4 &Vector4::euclid() {
	return (*this) = euclid(*this);
}