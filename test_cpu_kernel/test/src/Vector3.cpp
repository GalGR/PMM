#include "Vector3.h"

Vector3::operator Vector4() const { return Vector4(*this); }