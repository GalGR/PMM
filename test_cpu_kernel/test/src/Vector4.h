#pragma once

#include <cmath>

#define VECTOR_EPS 1.0E-10

struct Vector3;
#include "Vector3.h"

struct Vector4 {
	struct {
		double x;
		double y;
		double z;
		double w;
	};

	// Constructors
	Vector4(double x = 0.0, double y = 0.0, double z = 0.0, double w = 1.0) : x(x), y(y), z(z), w(w) {}
	Vector4(const Vector3 &vec3);

	// Get the Homogeneous coordinates
	double &operator [](size_t i) { return (&x)[i]; }
	const double &operator [](size_t i) const { return (&x)[i]; }

	// Get the Euclidean coordinates
	double euclid(size_t i) { return (&x)[i] / (&x)[3]; }
	double euclid(size_t i) const { return (&x)[i] / (&x)[3]; }
	double operator ()(size_t i) { return this->euclid(i); }
	double operator ()(size_t i) const { return this->euclid(i); }

	// Get the Euclidean vector
	static Vector3 euclid(const Vector4 &vector);
	Vector4 &euclid();

	// Cast to Vector3 (to euclid)
	operator Vector3() const;

	// Homogeneous Addition/Subtraction
	static Vector4 add(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = lhs[i] + rhs[i];
		}
		return vector;
	}
	static Vector4 sub(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = lhs[i] - rhs[i];
		}
		return vector;
	}
	friend Vector4 operator +(const Vector4 &lhs, const Vector4 &rhs) { return add(lhs, rhs); }
	friend Vector4 operator -(const Vector4 &lhs, const Vector4 &rhs) { return sub(lhs, rhs); }

	// Euclidean Addition/Subtraction
	static Vector4 euclid_add(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = lhs(i) + rhs(i);
		}
		return vector;
	}
	static Vector4 euclid_sub(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = lhs(i) - rhs(i);
		}
		return vector;
	}

	// Homogeneous Scalar Multiplication
	friend Vector4 operator *(double lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = lhs * rhs[i];
		}
		return vector;
	}
	friend Vector4 operator *(const Vector4 &lhs, double rhs) { return rhs * lhs; }

	// Euclidean Scalar Multiplication
	friend Vector4 operator &(double lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = lhs * rhs(i);
		}
		return vector;
	}
	friend Vector4 operator &(const Vector4 &lhs, double rhs) { return rhs & lhs; }

	// Euclidean Scalar Division
	friend Vector4 operator /(double lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = lhs / rhs[i];
		}
		return vector;
	}
	friend Vector4 operator /(const Vector4 &lhs, double rhs) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = lhs[i] / rhs;
		}
		return vector;
	}

	// Euclidean Scalar Division
	friend Vector4 operator |(double lhs, const Vector4 &rhs) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = lhs / rhs(i);
		}
		return vector;
	}
	friend Vector4 operator |(const Vector4 &lhs, double rhs) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = lhs(i) / rhs;
		}
		return vector;
	}

	// Homogeneous Dot Product
	friend double operator *(const Vector4 &lhs, const Vector4 &rhs) {
		double sum = 0;
		for (int i = 0; i < 4; ++i) {
			sum += lhs[i] * rhs[i];
		}
		return sum;
	}

	// Euclidean Dot Product
	friend double operator &(const Vector4 &lhs, const Vector4 &rhs) {
		double sum = 0;
		for (int i = 0; i < 3; ++i) {
			sum += lhs(i) * rhs(i);
		}
		return sum;
	}

	// Euclidean Cross Product
	friend Vector4 operator ^(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector;
		vector[0] = (lhs(1) * rhs(2)) - (lhs(2) * rhs(1));
		vector[1] = (lhs(2) * rhs(0)) - (lhs(0) * rhs(2));
		vector[2] = (lhs(0) * rhs(1)) - (lhs(1) * rhs(0));
		return vector;
	}

	// Vector Euclidean length
	double euclid_length() const {
		return sqrt((*this) & (*this));
	}

	// Vector Homogeneous length
	double length() const {
		return sqrt((*this) * (*this));
	}

	// Normal vector
	static Vector4 normal(const Vector4 &vector) {
		return vector | vector.euclid_length();
	}

	// Normalize vector
	Vector4 &normal() {
		return (*this) = normal(*this);
	}

	// Normal of two vectors
	static Vector4 normal(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 vector_ortho = rhs ^ lhs;
		return vector_ortho | vector_ortho.euclid_length();
	}
	friend Vector4 operator %(const Vector4 &lhs, const Vector4 &rhs) {
		return normal(lhs, rhs);
	}

	// Calculate a "from" "to" vector
	Vector4 to(const Vector4 &end) const {
		const Vector4 &start = *this;
		Vector4 vec;
		for (int i = 0; i < 3; ++i) {
			vec[i] = end(i) - start(i);
		}
		return vec;
	}
	Vector4 from(const Vector4 &start) const {
		const Vector4 &end = *this;
		Vector4 vec;
		for (int i = 0; i < 3; ++i) {
			vec[i] = end(i) - start(i);
		}
		return vec;
	}

	// Homogeneous vector equality
	friend bool equals(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 diff = sub(lhs, rhs);
		return diff[0] < VECTOR_EPS && diff[1] < VECTOR_EPS && diff[2] < VECTOR_EPS && diff[3] < VECTOR_EPS;
	}
	friend bool operator ==(const Vector4 &lhs, const Vector4 &rhs) { return equals(lhs, rhs); }

	// Euclidean vector equality
	static bool euclid_equals(const Vector4 &lhs, const Vector4 &rhs) {
		Vector4 diff = euclid_sub(lhs, rhs);
		return diff[0] < VECTOR_EPS && diff[1] < VECTOR_EPS && diff[2] < VECTOR_EPS;
	}

	// Homogeneous negation
	static Vector4 neg(const Vector4 &vec) {
		Vector4 vector;
		for (int i = 0; i < 4; ++i) {
			vector[i] = -vec[i];
		}
		return vector;
	}
	friend Vector4 operator -(const Vector4 &vec) { return neg(vec); }

	// Euclidean negation
	static Vector4 euclid_neg(const Vector4 &vec) {
		Vector4 vector;
		for (int i = 0; i < 3; ++i) {
			vector[i] = -vec(i);
		}
		return vector;
	}
	friend Vector4 operator ~(const Vector4 &vec) { return euclid_neg(vec); }
	
	// Homogeneous self negation
	Vector4 &neg() { return (*this) = neg(*this); }

	// Euclidean self negation
	Vector4 &euclid_neg() { return (*this) = euclid_neg(*this); }
};