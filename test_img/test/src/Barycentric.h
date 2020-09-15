#pragma once

#include "basetsd.h"
#include "Vector4.h"
#include "Vector3.h"
#include "Point.h"
#include <array>
#include <cmath>
#include <array>

class Bary {
private:
	std::array<double, 3> bary_;

	template <typename T>
	static inline double ROUND(T x) { return floor(x); }
	template <typename T>
	static inline int INT_ROUND(T x) { return (int)ROUND(x); }

	template <typename T>
	static double triangle_area_2(const std::array<Point_t<T>, 3> &poly) {
		Vector3 v01 = { (double)poly[1].x - poly[0].x, (double)poly[1].y - poly[0].y };
		Vector3 v02 = { (double)poly[2].x - poly[0].x, (double)poly[2].y - poly[0].y };
		return (v01 ^ v02).length();
	}
	template <typename T>
	static double triangle_area_2(const std::array<std::array<T, 2>, 3> &poly) {
		Vector3 v01 = { (double)poly[1][0] - poly[0][0], (double)poly[1][1] - poly[0][1] };
		Vector3 v02 = { (double)poly[2][0] - poly[0][0], (double)poly[2][1] - poly[0][1] };
		return (v01 ^ v02).length();
	}

	bool isPositive() const {
		return bary_[0] >= 0 && bary_[0] <= 1 && bary_[1] >= 0 && bary_[1] <= 1 && bary_[2] >= 0 && bary_[2] <= 1;
	}
	bool isNegative() const {
		return bary_[0] <= 0 && bary_[0] >= -1 && bary_[1] <= 0 && bary_[1] >= -1 && bary_[2] <= 0 && bary_[2] >= -1;
	}

public:
	inline const double &operator [](size_t i) const {
		return bary_[i];
	}
	inline double &operator [](size_t i) {
		return bary_[i];
	}

	Bary() = default;
	template <typename T>
	Bary(const Point_t<T> &p, const std::array<Point_t<T>, 3> &poly) {
		double b0 = (poly[1].y - poly[2].y) * p.x + (poly[2].x - poly[1].x) * p.y + (poly[1].x * poly[2].y - poly[1].y * poly[2].x);
		double b1 = (poly[2].y - poly[0].y) * p.x + (poly[0].x - poly[2].x) * p.y + (poly[2].x * poly[0].y - poly[2].y * poly[0].x);
		double b2 = (poly[0].y - poly[1].y) * p.x + (poly[1].x - poly[0].x) * p.y + (poly[0].x * poly[1].y - poly[0].y * poly[1].x);

		double area2 = triangle_area_2(poly);

		bary_[0] = b0 / area2;
		bary_[1] = b1 / area2;
		bary_[2] = b2 / area2;

		if (isNegative()) {
			bary_[0] = -bary_[0];
			bary_[1] = -bary_[1];
			bary_[2] = -bary_[2];
		}
	}
	template <typename T, typename V>
	Bary(const std::array<T, 2> &p, const std::array<std::array<V, 2>, 3> &poly) {
		double b0 = (poly[1][1] - poly[2][1]) * p[0] + (poly[2][0] - poly[1][0]) * p[1] + (poly[1][0] * poly[2][1] - poly[1][1] * poly[2][0]);
		double b1 = (poly[2][1] - poly[0][1]) * p[0] + (poly[0][0] - poly[2][0]) * p[1] + (poly[2][0] * poly[0][1] - poly[2][1] * poly[0][0]);
		double b2 = (poly[0][1] - poly[1][1]) * p[0] + (poly[1][0] - poly[0][0]) * p[1] + (poly[0][0] * poly[1][1] - poly[0][1] * poly[1][0]);

		double area2 = triangle_area_2(poly);

		bary_[0] = b0 / area2;
		bary_[1] = b1 / area2;
		bary_[2] = b2 / area2;

		if (isNegative()) {
			bary_[0] = -bary_[0];
			bary_[1] = -bary_[1];
			bary_[2] = -bary_[2];
		}
	}

	bool isInside() {
		return isPositive()/* || isNegative()*/;
	}

	Vector3 operator ()(const std::array<Vector4, 3> &vecs) {
		Vector3 mean;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				mean[j] += vecs[i](j) * bary_[i];
			}
		}
		return mean;
	}
	Vector3 operator ()(const std::array<Vector3, 3> &vecs) {
		Vector3 mean;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				mean[j] += vecs[i](j) * bary_[i];
			}
		}
		return mean;
	}
	template <typename T, size_t S>
	std::array<double, 3> operator ()(const std::array<std::array<T, S>, 3> &vecs) {
		std::array<double, S> mean;
		for (int i = 0; i < S; ++i) {
			mean[i] = 0;
		}
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < S; ++j) {
				mean[j] += vecs[i][j] * bary_[i];
			}
		}
		return mean;
	}
	template <typename T>
	Point_t<T> operator ()(const std::array<Point_t<T>, 3> &points) {
		Point_t<T> mean = { 0, 0 };
		for (int i = 0; i < 3; ++i) {
			mean.x += INT_ROUND(points[i].x * bary_[i]);
			mean.y += INT_ROUND(points[i].y * bary_[i]);
		}
		return mean;
	}
	double operator ()(const std::array<double, 3> &nums) {
		double mean = 0;
		for (int i = 0; i < 3; ++i) {
			mean += nums[i] * bary_[i];
		}
		return mean;
	}

	Vector3 operator ()(const Vector3 &vec) const {
		return Vector3{
			vec.x * bary_[0] + vec.x * bary_[1] + vec.x * bary_[2],
			vec.y * bary_[0] + vec.y * bary_[1] + vec.y * bary_[2],
			vec.z * bary_[0] + vec.z * bary_[1] + vec.z * bary_[2]
		};
	}
	template <typename T>
	Point_t<T> operator ()(const Point_t<T> &point) const {
		T x = point.x;
		T y = point.y;
		return Point_t<T>{
			x * bary_[0] + x * bary_[1] + x * bary_[2],
			y * bary_[0] + y * bary_[1] + y * bary_[2]
		};
	}
	Point_t<int> operator ()(const Point_t<int> &point) const {
		int x = point.x;
		int y = point.y;
		return Point_t<int>{
			INT_ROUND(x * bary_[0] + x * bary_[1] + x * bary_[2]),
			INT_ROUND(y * bary_[0] + y * bary_[1] + y * bary_[2])
		};
	}
	template <typename T>
	std::array<double, 2> operator ()(const std::array<T, 2> &point) const {
		double x = point[0];
		double y = point[1];
		return std::array<double, 2>{
			x * bary_[0] + x * bary_[1] + x * bary_[2],
			y * bary_[0] + y * bary_[1] + y * bary_[2]
		};
	}
};