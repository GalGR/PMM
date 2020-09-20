#pragma once

template <typename Pos_T>
struct Point_t {
	Pos_T x;
	Pos_T y;

	inline Point_t() = default;
	inline Point_t(Pos_T x, Pos_T y) : x(x), y(y) {}
	inline const Pos_T &operator[](const short i) const { return (&x)[i]; }
	inline Pos_T &operator[](const short i) { return (&x)[i]; }
	inline friend Point_t<Pos_T> operator+(const Point_t &lhs, const Point_t &rhs) { return Point_t(lhs.x + rhs.x, lhs.y + rhs.y); }
	inline friend Point_t<Pos_T> operator-(const Point_t &lhs, const Point_t &rhs) { return Point_t(lhs.x - rhs.x, lhs.y - rhs.y); }
};

typedef Point_t<int> PointI;
typedef Point_t<double> PointD;