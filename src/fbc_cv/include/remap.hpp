// fbc_cv is free software and uses the same licence as OpenCV
// Email: fengbingchun@163.com

#ifndef FBC_CV_REMAP_HPP_
#define FBC_CV_REMAP_HPP_

/* reference: include/opencv2/imgproc.hpp
              modules/imgproc/src/imgwarp.cpp
*/

#include <typeinfo>
#include "core/mat.hpp"
#include "core/base.hpp"
#include "core/core.hpp"
#include "imgproc.hpp"
#include "resize.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace fbc {

const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];

template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3> static int remap_nearest(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue);
template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3> static int remap_linear(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue);
template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3> static int remap_cubic(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue);
template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3> static int remap_lanczos4(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue);

// Applies a generic geometrical transformation to an image
// transforms the source image using the specified map, this function cannot operate in-place
/*
\f[\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\f]
*/
// support type: uchar/float
template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
int remap(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst, const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2,
	int interpolation, int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	FBC_Assert(map1.size().area() > 0);
	FBC_Assert(map2.empty() || map1.size() == map2.size());
	FBC_Assert(typeid(float).name() == typeid(_Tp2).name() || typeid(short).name() == typeid(_Tp2).name());
	FBC_Assert(typeid(float).name() == typeid(_Tp3).name() || typeid(short).name() == typeid(_Tp3).name() || typeid(ushort).name() == typeid(_Tp3).name());
	if (typeid(short).name() == typeid(_Tp2).name() && chs2 == 2) {
		FBC_Assert(((typeid(short).name() == typeid(_Tp3).name() || typeid(ushort).name() == typeid(_Tp3).name()) && chs3 == 1) || map2.empty());
	} else {
		FBC_Assert(((typeid(float).name() == typeid(_Tp2).name() || typeid(short).name() == typeid(_Tp2).name()) && chs2 == 2 && map2.empty()) ||
			(typeid(float).name() == typeid(_Tp2).name() && typeid(float).name() == typeid(_Tp3).name() && chs2 == chs3 && chs2 == 1));
	}
	FBC_Assert(map2.empty() || map2.size() == map1.size());
	FBC_Assert(src.data != dst.data);
	FBC_Assert(typeid(uchar).name() == typeid(_Tp1).name() || typeid(float).name() == typeid(_Tp1).name()); // uchar || float

	switch (interpolation) {
		case 0: {
			remap_nearest(src, dst, map1, map2, borderMode, borderValue);
			break;
		}
		case 1:
		case 3: {
			remap_linear(src, dst, map1, map2, borderMode, borderValue);
			break;
		}
		case 2: {
			remap_cubic(src, dst, map1, map2, borderMode, borderValue);
			break;
		}
		case 4: {
			remap_lanczos4(src, dst, map1, map2, borderMode, borderValue);
			break;
		}
		default:
			return -1;
	}

	return 0;
}

template<typename _Tp>
static inline void interpolateLinear(_Tp x, _Tp* coeffs)
{
	coeffs[0] = 1.f - x;
	coeffs[1] = x;
}

template<typename _Tp>
static void initInterTab1D(int method, float* tab, int tabsz)
{
	float scale = 1.f / tabsz;
	if (method == INTER_LINEAR) {
		for (int i = 0; i < tabsz; i++, tab += 2)
			interpolateLinear<float>(i*scale, tab);
	} else if (method == INTER_CUBIC) {
		for (int i = 0; i < tabsz; i++, tab += 4)
			interpolateCubic<float>(i*scale, tab);
	} else if (method == INTER_LANCZOS4) {
		for (int i = 0; i < tabsz; i++, tab += 8)
			interpolateLanczos4<float>(i*scale, tab);
	} else {
		FBC_Error("Unknown interpolation method");
	}
}

template<typename _Tp>
static const void* initInterTab2D(int method, bool fixpt)
{
	static bool inittab[INTER_MAX + 1] = { false };
	float* tab = 0;
	short* itab = 0;
	int ksize = 0;
	if (method == INTER_LINEAR) {
		tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;
	} else if (method == INTER_CUBIC) {
		tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize = 4;
	} else if (method == INTER_LANCZOS4) {
		tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize = 8;
	} else {
		FBC_Error("Unknown/unsupported interpolation type");
	}

	if (!inittab[method]) {
		AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
		int i, j, k1, k2;
		initInterTab1D<float>(method, _tab, INTER_TAB_SIZE);
		for (i = 0; i < INTER_TAB_SIZE; i++) {
			for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize*ksize, itab += ksize*ksize) {
				int isum = 0;
				NNDeltaTab_i[i*INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
				NNDeltaTab_i[i*INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;

				for (k1 = 0; k1 < ksize; k1++) {
					float vy = _tab[i*ksize + k1];
					for (k2 = 0; k2 < ksize; k2++) {
						float v = vy*_tab[j*ksize + k2];
						tab[k1*ksize + k2] = v;
						isum += itab[k1*ksize + k2] = saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
					}
				}

				if (isum != INTER_REMAP_COEF_SCALE) {
					int diff = isum - INTER_REMAP_COEF_SCALE;
					int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
					for (k1 = ksize2; k1 < ksize2 + 2; k1++) {
						for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
							if (itab[k1*ksize + k2] < itab[mk1*ksize + mk2])
								mk1 = k1, mk2 = k2;
							else if (itab[k1*ksize + k2] > itab[Mk1*ksize + Mk2])
								Mk1 = k1, Mk2 = k2;
						}
					}
					if (diff < 0)
						itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
					else
						itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
				}
			}
		}
		tab -= INTER_TAB_SIZE2*ksize*ksize;
		itab -= INTER_TAB_SIZE2*ksize*ksize;
		inittab[method] = true;
	}

	return fixpt ? (const void*)itab : (const void*)tab;
}

template<typename _Tp>
static bool initAllInterTab2D()
{
	return  initInterTab2D<uchar>(INTER_LINEAR, false) &&
		initInterTab2D<uchar>(INTER_LINEAR, true) &&
		initInterTab2D<uchar>(INTER_CUBIC, false) &&
		initInterTab2D<uchar>(INTER_CUBIC, true) &&
		initInterTab2D<uchar>(INTER_LANCZOS4, false) &&
		initInterTab2D<uchar>(INTER_LANCZOS4, true);
}

static volatile bool doInitAllInterTab2D = initAllInterTab2D<uchar>();

template<typename _Tp1, typename _Tp2, int chs1, int chs2>
static void remapNearest(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst, const Mat_<_Tp2, chs2>& _xy, int borderType, const Scalar& _borderValue)
{
	Size ssize = _src.size(), dsize = _dst.size();
	int cn = _src.channels;
	const _Tp1* S0 = (const _Tp1*)_src.ptr();
	size_t sstep = _src.step / sizeof(S0[0]);
	Scalar_<_Tp1> cval(saturate_cast<_Tp1>(_borderValue[0]), saturate_cast<_Tp1>(_borderValue[1]), saturate_cast<_Tp1>(_borderValue[2]), saturate_cast<_Tp1>(_borderValue[3]));
	int dx, dy;

	unsigned width1 = ssize.width, height1 = ssize.height;

	for (dy = 0; dy < dsize.height; dy++) {
		_Tp1* D = (_Tp1*)_dst.ptr(dy);
		const short* XY = (const short*)_xy.ptr(dy);

		if (cn == 1) {
			for (dx = 0; dx < dsize.width; dx++) {
				int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
				if ((unsigned)sx < width1 && (unsigned)sy < height1) {
					D[dx] = S0[sy*sstep + sx];
				} else {
					if (borderType == BORDER_REPLICATE) {
						sx = clip<int>(sx, 0, ssize.width);
						sy = clip<int>(sy, 0, ssize.height);
						D[dx] = S0[sy*sstep + sx];
					} else if (borderType == BORDER_CONSTANT) {
						D[dx] = cval[0];
					} else if (borderType != BORDER_TRANSPARENT) {
						sx = borderInterpolate<int>(sx, ssize.width, borderType);
						sy = borderInterpolate<int>(sy, ssize.height, borderType);
						D[dx] = S0[sy*sstep + sx];
					}
				}
			}
		} else {
			for (dx = 0; dx < dsize.width; dx++, D += cn) {
				int sx = XY[dx * 2], sy = XY[dx * 2 + 1], k;
				const _Tp1 *S;
				if ((unsigned)sx < width1 && (unsigned)sy < height1) {
					if (cn == 3) {
						S = S0 + sy*sstep + sx * 3;
						D[0] = S[0], D[1] = S[1], D[2] = S[2];
					} else if (cn == 4) {
						S = S0 + sy*sstep + sx * 4;
						D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
					} else {
						S = S0 + sy*sstep + sx*cn;
						for (k = 0; k < cn; k++)
							D[k] = S[k];
					}
				} else if (borderType != BORDER_TRANSPARENT) {
					if (borderType == BORDER_REPLICATE) {
						sx = clip<int>(sx, 0, ssize.width);
						sy = clip<int>(sy, 0, ssize.height);
						S = S0 + sy*sstep + sx*cn;
					} else if (borderType == BORDER_CONSTANT) {
						S = &cval[0];
					} else {
						sx = borderInterpolate<int>(sx, ssize.width, borderType);
						sy = borderInterpolate<int>(sy, ssize.height, borderType);
						S = S0 + sy*sstep + sx*cn;
					}
					for (k = 0; k < cn; k++)
						D[k] = S[k];
				}
			}
		}
	}
}

template<class CastOp, typename AT, typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remapBilinear(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst,
	const Mat_<_Tp2, chs2>& _xy, const Mat_<_Tp3, chs3>& _fxy, const void* _wtab, int borderType, const Scalar& _borderValue)
{
	typedef typename CastOp::rtype T; 
	typedef typename CastOp::type1 WT;
	Size ssize = _src.size(), dsize = _dst.size();
	int k, cn = _src.channels;
	const AT* wtab = (const AT*)_wtab;
	const T* S0 = (const T*)_src.ptr();
	size_t sstep = _src.step / sizeof(S0[0]);
	T cval[FBC_CN_MAX];
	int dx, dy;
	CastOp castOp;

	for (k = 0; k < cn; k++)
		cval[k] = saturate_cast<T>(_borderValue[k & 3]);

	unsigned width1 = std::max(ssize.width - 1, 0), height1 = std::max(ssize.height - 1, 0);
	FBC_Assert(ssize.area() > 0);

	for (dy = 0; dy < dsize.height; dy++) {
		T* D = (T*)_dst.ptr(dy);
		const short* XY = (const short*)_xy.ptr(dy);
		const ushort* FXY = (const ushort*)_fxy.ptr(dy);
		int X0 = 0;
		bool prevInlier = false;

		for (dx = 0; dx <= dsize.width; dx++) {
			bool curInlier = dx < dsize.width ? (unsigned)XY[dx * 2] < width1 && (unsigned)XY[dx * 2 + 1] < height1 : !prevInlier;
			if (curInlier == prevInlier)
				continue;

			int X1 = dx;
			dx = X0;
			X0 = X1;
			prevInlier = curInlier;

			if (!curInlier) {
				int len = 0;
				D += len*cn;
				dx += len;

				if (cn == 1) {
					for (; dx < X1; dx++, D++) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy*sstep + sx;
						*D = castOp(WT(S[0] * w[0] + S[1] * w[1] + S[sstep] * w[2] + S[sstep + 1] * w[3]));
					}
				} else if (cn == 2) {
					for (; dx < X1; dx++, D += 2) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy*sstep + sx * 2;
						WT t0 = S[0] * w[0] + S[2] * w[1] + S[sstep] * w[2] + S[sstep + 2] * w[3];
						WT t1 = S[1] * w[0] + S[3] * w[1] + S[sstep + 1] * w[2] + S[sstep + 3] * w[3];
						D[0] = castOp(t0); D[1] = castOp(t1);
					}
				} else if (cn == 3) {

					for (; dx < X1 - 4; dx += 4) 
					{
#if __ARM_NEON
						int sx0 = *(XY + ((dx + 0) * 2));
						int sx1 = *(XY + ((dx + 1) * 2));
						int sx2 = *(XY + ((dx + 2) * 2));
						int sx3 = *(XY + ((dx + 3) * 2));
						
						int sy0 = *(XY + ((dx + 0) * 2 + 1));
						int sy1 = *(XY + ((dx + 1) * 2 + 1));
						int sy2 = *(XY + ((dx + 2) * 2 + 1));
						int sy3 = *(XY + ((dx + 3) * 2 + 1));

						const AT* w0 = wtab + *(FXY + dx + 0) * 4;
						const AT* w1 = wtab + *(FXY + dx + 1) * 4;
						const AT* w2 = wtab + *(FXY + dx + 2) * 4;
						const AT* w3 = wtab + *(FXY + dx + 3) * 4;

						const T* S_0 = S0 + sy0 *sstep + sx0 * 3;
						const T* S_1 = S0 + sy1 *sstep + sx1 * 3;
						const T* S_2 = S0 + sy2 *sstep + sx2 * 3;
						const T* S_3 = S0 + sy3 *sstep + sx3 * 3;	

						WT t0_0 = S_0[0] * w0[0] + S_0[3] * w0[1] + S_0[sstep + 0] * w0[2] + S_0[sstep + 3] * w0[3];
						WT t0_1 = S_0[1] * w0[0] + S_0[4] * w0[1] + S_0[sstep + 1] * w0[2] + S_0[sstep + 4] * w0[3];
						WT t0_2 = S_0[2] * w0[0] + S_0[5] * w0[1] + S_0[sstep + 2] * w0[2] + S_0[sstep + 5] * w0[3];

						WT t1_0 = S_1[0] * w1[0] + S_1[3] * w1[1] + S_1[sstep + 0] * w1[2] + S_1[sstep + 3] * w1[3];
						WT t1_1 = S_1[1] * w1[0] + S_1[4] * w1[1] + S_1[sstep + 1] * w1[2] + S_1[sstep + 4] * w1[3];
						WT t1_2 = S_1[2] * w1[0] + S_1[5] * w1[1] + S_1[sstep + 2] * w1[2] + S_1[sstep + 5] * w1[3];

						WT t2_0 = S_2[0] * w2[0] + S_2[3] * w2[1] + S_2[sstep + 0] * w2[2] + S_2[sstep + 3] * w2[3];
						WT t2_1 = S_2[1] * w2[0] + S_2[4] * w2[1] + S_2[sstep + 1] * w2[2] + S_2[sstep + 4] * w2[3];
						WT t2_2 = S_2[2] * w2[0] + S_2[5] * w2[1] + S_2[sstep + 2] * w2[2] + S_2[sstep + 5] * w2[3];

						WT t3_0 = S_3[0] * w3[0] + S_3[3] * w3[1] + S_3[sstep + 0] * w3[2] + S_3[sstep + 3] * w3[3];
						WT t3_1 = S_3[1] * w3[0] + S_3[4] * w3[1] + S_3[sstep + 1] * w3[2] + S_3[sstep + 4] * w3[3];
						WT t3_2 = S_3[2] * w3[0] + S_3[5] * w3[1] + S_3[sstep + 2] * w3[2] + S_3[sstep + 5] * w3[3];

						D[0] = castOp(t0_0);
						D[1] = castOp(t0_1);
						D[2] = castOp(t0_2);

						D[3] = castOp(t1_0);
						D[4] = castOp(t1_1);
						D[5] = castOp(t1_2);

						D[6] = castOp(t2_0);
						D[7] = castOp(t2_1);
						D[8] = castOp(t2_2);

						D[9] = castOp(t3_0);
						D[10] = castOp(t3_1);
						D[11] = castOp(t3_2);
#else // __ARM_NEON
						int sx0 = *(XY + ((dx + 0) * 2));
						int sx1 = *(XY + ((dx + 1) * 2));
						int sx2 = *(XY + ((dx + 2) * 2));
						int sx3 = *(XY + ((dx + 3) * 2));
						
						int sy0 = *(XY + ((dx + 0) * 2 + 1));
						int sy1 = *(XY + ((dx + 1) * 2 + 1));
						int sy2 = *(XY + ((dx + 2) * 2 + 1));
						int sy3 = *(XY + ((dx + 3) * 2 + 1));

						const AT* w0 = wtab + *(FXY + dx + 0) * 4;
						const AT* w1 = wtab + *(FXY + dx + 1) * 4;
						const AT* w2 = wtab + *(FXY + dx + 2) * 4;
						const AT* w3 = wtab + *(FXY + dx + 3) * 4;

						const T* S_0 = S0 + sy0 *sstep + sx0 * 3;
						const T* S_1 = S0 + sy1 *sstep + sx1 * 3;
						const T* S_2 = S0 + sy2 *sstep + sx2 * 3;
						const T* S_3 = S0 + sy3 *sstep + sx3 * 3;

						WT t0_0 = S_0[0] * w0[0] + S_0[3] * w0[1] + S_0[sstep + 0] * w0[2] + S_0[sstep + 3] * w0[3];
						WT t0_1 = S_0[1] * w0[0] + S_0[4] * w0[1] + S_0[sstep + 1] * w0[2] + S_0[sstep + 4] * w0[3];
						WT t0_2 = S_0[2] * w0[0] + S_0[5] * w0[1] + S_0[sstep + 2] * w0[2] + S_0[sstep + 5] * w0[3];

						WT t1_0 = S_1[0] * w1[0] + S_1[3] * w1[1] + S_1[sstep + 0] * w1[2] + S_1[sstep + 3] * w1[3];
						WT t1_1 = S_1[1] * w1[0] + S_1[4] * w1[1] + S_1[sstep + 1] * w1[2] + S_1[sstep + 4] * w1[3];
						WT t1_2 = S_1[2] * w1[0] + S_1[5] * w1[1] + S_1[sstep + 2] * w1[2] + S_1[sstep + 5] * w1[3];

						WT t2_0 = S_2[0] * w2[0] + S_2[3] * w2[1] + S_2[sstep + 0] * w2[2] + S_2[sstep + 3] * w2[3];
						WT t2_1 = S_2[1] * w2[0] + S_2[4] * w2[1] + S_2[sstep + 1] * w2[2] + S_2[sstep + 4] * w2[3];
						WT t2_2 = S_2[2] * w2[0] + S_2[5] * w2[1] + S_2[sstep + 2] * w2[2] + S_2[sstep + 5] * w2[3];

						WT t3_0 = S_3[0] * w3[0] + S_3[3] * w3[1] + S_3[sstep + 0] * w3[2] + S_3[sstep + 3] * w3[3];
						WT t3_1 = S_3[1] * w3[0] + S_3[4] * w3[1] + S_3[sstep + 1] * w3[2] + S_3[sstep + 4] * w3[3];
						WT t3_2 = S_3[2] * w3[0] + S_3[5] * w3[1] + S_3[sstep + 2] * w3[2] + S_3[sstep + 5] * w3[3];

						D[0] = castOp(t0_0);
						D[1] = castOp(t0_1);
						D[2] = castOp(t0_2);

						D[3] = castOp(t1_0);
						D[4] = castOp(t1_1);
						D[5] = castOp(t1_2);

						D[6] = castOp(t2_0);
						D[7] = castOp(t2_1);
						D[8] = castOp(t2_2);

						D[9] = castOp(t3_0);
						D[10] = castOp(t3_1);
						D[11] = castOp(t3_2);
#endif
						D += 12;
					}

					for (; dx < X1; dx++) 
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy *sstep + sx * 3;

						WT t0 = S[0] * w[0] + S[3] * w[1] + S[sstep + 0] * w[2] + S[sstep + 3] * w[3];
						WT t1 = S[1] * w[0] + S[4] * w[1] + S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
						WT t2 = S[2] * w[0] + S[5] * w[1] + S[sstep + 2] * w[2] + S[sstep + 5] * w[3];

						D[0] = castOp(t0); 
						D[1] = castOp(t1);
						D[2] = castOp(t2);

						D += 3;
					}
				} else if (cn == 4) {
					for (; dx < X1; dx++, D += 4) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy*sstep + sx * 4;
						WT t0 = S[0] * w[0] + S[4] * w[1] + S[sstep] * w[2] + S[sstep + 4] * w[3];
						WT t1 = S[1] * w[0] + S[5] * w[1] + S[sstep + 1] * w[2] + S[sstep + 5] * w[3];
						D[0] = castOp(t0); D[1] = castOp(t1);
						t0 = S[2] * w[0] + S[6] * w[1] + S[sstep + 2] * w[2] + S[sstep + 6] * w[3];
						t1 = S[3] * w[0] + S[7] * w[1] + S[sstep + 3] * w[2] + S[sstep + 7] * w[3];
						D[2] = castOp(t0); D[3] = castOp(t1);
					}
				} else {
					for (; dx < X1; dx++, D += cn) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy*sstep + sx*cn;
						for (k = 0; k < cn; k++) {
							WT t0 = S[k] * w[0] + S[k + cn] * w[1] + S[sstep + k] * w[2] + S[sstep + k + cn] * w[3];
							D[k] = castOp(t0);
						}
					}
				}
			} else {
				if (borderType == BORDER_TRANSPARENT && cn != 3) {
					D += (X1 - dx)*cn;
					dx = X1;
					continue;
				}

				if (cn == 1) {
					for (; dx < X1; dx++, D++) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
							D[0] = cval[0];
						} else {
							int sx0, sx1, sy0, sy1;
							T v0, v1, v2, v3;
							const AT* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE) {
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0[sy0*sstep + sx0];
								v1 = S0[sy0*sstep + sx1];
								v2 = S0[sy1*sstep + sx0];
								v3 = S0[sy1*sstep + sx1];
							} else {
								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
							}
							D[0] = castOp(WT(v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]));
						}
					}
				} else {
					for (; dx < X1; dx++, D += cn) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
							for (k = 0; k < cn; k++)
								D[k] = cval[k];
						} else {
							int sx0, sx1, sy0, sy1;
							const T *v0, *v1, *v2, *v3;
							const AT* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE) {
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0 + sy0*sstep + sx0*cn;
								v1 = S0 + sy0*sstep + sx1*cn;
								v2 = S0 + sy1*sstep + sx0*cn;
								v3 = S0 + sy1*sstep + sx1*cn;
							} else if (borderType == BORDER_TRANSPARENT && ((unsigned)sx >= (unsigned)(ssize.width - 1) || (unsigned)sy >= (unsigned)(ssize.height - 1))) {
								continue;
							} else {
								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
							}
							for (k = 0; k < cn; k++)
								D[k] = castOp(WT(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3]));
						}
					}
				}
			}
		}
	}

	return 0;
}

static inline unsigned char castOp_tq(int src)
{
	return saturate_cast<unsigned char>((src + 16384) >> 15);
}

template<class CastOp, typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remapBilinear_fixed(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst,
	const Mat_<_Tp2, chs2>& _xy, const Mat_<_Tp3, chs3>& _fxy, const void* _wtab, int borderType, const Scalar& _borderValue)
{
	//typedef typename CastOp::rtype T; //uchar
	//typedef typename CastOp::type1 WT;//int
	Size ssize = _src.size(), dsize = _dst.size();
	int k, cn = _src.channels;
	const short* wtab = (const short*)_wtab; //short
	const uchar* S0 = (const uchar*)_src.ptr();
	size_t sstep = _src.step / sizeof(S0[0]);
	uchar cval[FBC_CN_MAX];
	int dx, dy;
	CastOp castOp;

	for (k = 0; k < cn; k++)
		cval[k] = saturate_cast<uchar>(_borderValue[k & 3]);

	unsigned width1 = std::max(ssize.width - 1, 0), height1 = std::max(ssize.height - 1, 0);
	FBC_Assert(ssize.area() > 0);

	for (dy = 0; dy < dsize.height; dy++) {
		uchar* D = (uchar*)_dst.ptr(dy);
		const short* XY = (const short*)_xy.ptr(dy);
		const ushort* FXY = (const ushort*)_fxy.ptr(dy);
		int X0 = 0;
		bool prevInlier = false;

		for (dx = 0; dx <= dsize.width; dx++) {
			bool curInlier = dx < dsize.width ? (unsigned)XY[dx * 2] < width1 && (unsigned)XY[dx * 2 + 1] < height1 : !prevInlier;
			if (curInlier == prevInlier)
				continue;

			int X1 = dx;
			dx = X0;
			X0 = X1;
			prevInlier = curInlier;

			if (!curInlier) {
				int len = 0;
				D += len*cn;
				dx += len;

				if (cn == 1) {
					for (; dx < X1; dx++, D++) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const short* w = wtab + FXY[dx] * 4;
						const uchar* S = S0 + sy*sstep + sx;
						*D = castOp_tq(int(S[0] * w[0] + S[1] * w[1] + S[sstep] * w[2] + S[sstep + 1] * w[3]));
					}
				} else if (cn == 2) {
					for (; dx < X1; dx++, D += 2) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const short* w = wtab + FXY[dx] * 4;
						const uchar* S = S0 + sy*sstep + sx * 2;
						int t0 = S[0] * w[0] + S[2] * w[1] + S[sstep] * w[2] + S[sstep + 2] * w[3];
						int t1 = S[1] * w[0] + S[3] * w[1] + S[sstep + 1] * w[2] + S[sstep + 3] * w[3];
						D[0] = castOp_tq(t0); 
						D[1] = castOp_tq(t1);
					}
				} else if (cn == 3) {

					for (; dx < X1 - 4; dx += 4) 
					{
#if __ARM_NEON
						int sx0 = *(XY + ((dx + 0) * 2));
						int sx1 = *(XY + ((dx + 1) * 2));
						int sx2 = *(XY + ((dx + 2) * 2));
						int sx3 = *(XY + ((dx + 3) * 2));
						
						int sy0 = *(XY + ((dx + 0) * 2 + 1));
						int sy1 = *(XY + ((dx + 1) * 2 + 1));
						int sy2 = *(XY + ((dx + 2) * 2 + 1));
						int sy3 = *(XY + ((dx + 3) * 2 + 1));

						const uchar* S_0 = S0 + sy0 *sstep + sx0 * 3;
						const uchar* S_1 = S0 + sy1 *sstep + sx1 * 3;
						const uchar* S_2 = S0 + sy2 *sstep + sx2 * 3;
						const uchar* S_3 = S0 + sy3 *sstep + sx3 * 3;

						const short* w0 = wtab + *(FXY + dx + 0) * 4;
						const short* w1 = wtab + *(FXY + dx + 1) * 4;
						const short* w2 = wtab + *(FXY + dx + 2) * 4;
						const short* w3 = wtab + *(FXY + dx + 3) * 4;						

						uint8x8_t _S_0 = vld1_u8(S_0);
						uint8x8_t _S_0_sstep = vld1_u8(S_0 + sstep);
						int16x4_t _w0 = vld1_s16(w0);

						uint8x8_t _S_1 = vld1_u8(S_1);
						uint8x8_t _S_1_sstep = vld1_u8(S_1 + sstep);
						int16x4_t _w1 = vld1_s16(w1);

						uint8x8_t _S_2 = vld1_u8(S_2);
						uint8x8_t _S_2_sstep = vld1_u8(S_2 + sstep);
						int16x4_t _w2 = vld1_s16(w2);

						uint8x8_t _S_3 = vld1_u8(S_3);
						uint8x8_t _S_3_sstep = vld1_u8(S_3 + sstep);
						int16x4_t _w3 = vld1_s16(w3);

						int16x8_t _S_0_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_0));
						int16x8_t _S_0_sstep_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_0_sstep));
						int16x8_t _S_1_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_1));
						int16x8_t _S_1_sstep_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_1_sstep));
						int16x8_t _S_2_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_2));
						int16x8_t _S_2_sstep_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_2_sstep));
						int16x8_t _S_3_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_3));
						int16x8_t _S_3_sstep_s16 = vreinterpretq_s16_u16(vmovl_u8(_S_3_sstep));

						int16x4_t _S_0_s16n = vext_s16(vget_low_s16(_S_0_s16), vget_high_s16(_S_0_s16), 3);
						int16x4_t _S_0_sstep_s16n = vext_s16(vget_low_s16(_S_0_sstep_s16), vget_high_s16(_S_0_sstep_s16), 3);
						int16x4_t _S_1_s16n = vext_s16(vget_low_s16(_S_1_s16), vget_high_s16(_S_1_s16), 3);
						int16x4_t _S_1_sstep_s16n = vext_s16(vget_low_s16(_S_1_sstep_s16), vget_high_s16(_S_1_sstep_s16), 3);
						int16x4_t _S_2_s16n = vext_s16(vget_low_s16(_S_2_s16), vget_high_s16(_S_2_s16), 3);
						int16x4_t _S_2_sstep_s16n = vext_s16(vget_low_s16(_S_2_sstep_s16), vget_high_s16(_S_2_sstep_s16), 3);
						int16x4_t _S_3_s16n = vext_s16(vget_low_s16(_S_3_s16), vget_high_s16(_S_3_s16), 3);
						int16x4_t _S_3_sstep_s16n = vext_s16(vget_low_s16(_S_3_sstep_s16), vget_high_s16(_S_3_sstep_s16), 3);

						int32x4_t _t0 = vmull_lane_s16(vget_low_s16(_S_0_s16), _w0, 0);
						int32x4_t _t1 = vmull_lane_s16(vget_low_s16(_S_1_s16), _w1, 0);
						int32x4_t _t2 = vmull_lane_s16(vget_low_s16(_S_2_s16), _w2, 0);
						int32x4_t _t3 = vmull_lane_s16(vget_low_s16(_S_3_s16), _w3, 0);

						_t0 = vmlal_lane_s16(_t0, _S_0_s16n, _w0, 1);
						_t1 = vmlal_lane_s16(_t1, _S_1_s16n, _w1, 1);
						_t2 = vmlal_lane_s16(_t2, _S_2_s16n, _w2, 1);
						_t3 = vmlal_lane_s16(_t3, _S_3_s16n, _w3, 1);

						_t0 = vmlal_lane_s16(_t0, vget_low_s16(_S_0_sstep_s16), _w0, 2);
						_t1 = vmlal_lane_s16(_t1, vget_low_s16(_S_1_sstep_s16), _w1, 2);
						_t2 = vmlal_lane_s16(_t2, vget_low_s16(_S_2_sstep_s16), _w2, 2);
						_t3 = vmlal_lane_s16(_t3, vget_low_s16(_S_3_sstep_s16), _w3, 2);

						_t0 = vmlal_lane_s16(_t0, _S_0_sstep_s16n, _w0, 3);
						_t1 = vmlal_lane_s16(_t1, _S_1_sstep_s16n, _w1, 3);
						_t2 = vmlal_lane_s16(_t2, _S_2_sstep_s16n, _w2, 3);
						_t3 = vmlal_lane_s16(_t3, _S_3_sstep_s16n, _w3, 3);

						int32x4_t _tmp = vdupq_n_s32(16384);

						_t0 = vshrq_n_s32(vaddq_s32(_t0, _tmp), 15);
						_t1 = vshrq_n_s32(vaddq_s32(_t1, _tmp), 15);
						_t2 = vshrq_n_s32(vaddq_s32(_t2, _tmp), 15);
						_t3 = vshrq_n_s32(vaddq_s32(_t3, _tmp), 15);

						uint16x8_t _d0_u16 = vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(_t0), vmovn_s32(_t1)));
						uint16x8_t _d1_u16 = vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(_t2), vmovn_s32(_t3)));
						uint8x16_t _d_u8 = vcombine_u8(vmovn_u16(_d0_u16), vmovn_u16(_d1_u16));

						D[0] = _d_u8[0];
						D[1] = _d_u8[1];
						D[2] = _d_u8[2];

						D[3] = _d_u8[4];
						D[4] = _d_u8[5];
						D[5] = _d_u8[6];

						D[6] = _d_u8[8];
						D[7] = _d_u8[9];
						D[8] = _d_u8[10];

						D[9]  = _d_u8[12];
						D[10] = _d_u8[13];
						D[11] = _d_u8[14];					
#else						
						int sx0 = *(XY + ((dx + 0) * 2));
						int sx1 = *(XY + ((dx + 1) * 2));
						int sx2 = *(XY + ((dx + 2) * 2));
						int sx3 = *(XY + ((dx + 3) * 2));
						
						int sy0 = *(XY + ((dx + 0) * 2 + 1));
						int sy1 = *(XY + ((dx + 1) * 2 + 1));
						int sy2 = *(XY + ((dx + 2) * 2 + 1));
						int sy3 = *(XY + ((dx + 3) * 2 + 1));

						const short* w0 = wtab + *(FXY + dx + 0) * 4;
						const short* w1 = wtab + *(FXY + dx + 1) * 4;
						const short* w2 = wtab + *(FXY + dx + 2) * 4;
						const short* w3 = wtab + *(FXY + dx + 3) * 4;

						const uchar* S_0 = S0 + sy0 *sstep + sx0 * 3;
						const uchar* S_1 = S0 + sy1 *sstep + sx1 * 3;
						const uchar* S_2 = S0 + sy2 *sstep + sx2 * 3;
						const uchar* S_3 = S0 + sy3 *sstep + sx3 * 3;

						int t0_0 = S_0[0] * w0[0] + S_0[3] * w0[1] + S_0[sstep + 0] * w0[2] + S_0[sstep + 3] * w0[3];
						int t0_1 = S_0[1] * w0[0] + S_0[4] * w0[1] + S_0[sstep + 1] * w0[2] + S_0[sstep + 4] * w0[3];
						int t0_2 = S_0[2] * w0[0] + S_0[5] * w0[1] + S_0[sstep + 2] * w0[2] + S_0[sstep + 5] * w0[3];

						int t1_0 = S_1[0] * w1[0] + S_1[3] * w1[1] + S_1[sstep + 0] * w1[2] + S_1[sstep + 3] * w1[3];
						int t1_1 = S_1[1] * w1[0] + S_1[4] * w1[1] + S_1[sstep + 1] * w1[2] + S_1[sstep + 4] * w1[3];
						int t1_2 = S_1[2] * w1[0] + S_1[5] * w1[1] + S_1[sstep + 2] * w1[2] + S_1[sstep + 5] * w1[3];

						int t2_0 = S_2[0] * w2[0] + S_2[3] * w2[1] + S_2[sstep + 0] * w2[2] + S_2[sstep + 3] * w2[3];
						int t2_1 = S_2[1] * w2[0] + S_2[4] * w2[1] + S_2[sstep + 1] * w2[2] + S_2[sstep + 4] * w2[3];
						int t2_2 = S_2[2] * w2[0] + S_2[5] * w2[1] + S_2[sstep + 2] * w2[2] + S_2[sstep + 5] * w2[3];

						int t3_0 = S_3[0] * w3[0] + S_3[3] * w3[1] + S_3[sstep + 0] * w3[2] + S_3[sstep + 3] * w3[3];
						int t3_1 = S_3[1] * w3[0] + S_3[4] * w3[1] + S_3[sstep + 1] * w3[2] + S_3[sstep + 4] * w3[3];
						int t3_2 = S_3[2] * w3[0] + S_3[5] * w3[1] + S_3[sstep + 2] * w3[2] + S_3[sstep + 5] * w3[3];

						D[0] = castOp(t0_0);
						D[1] = castOp(t0_1);
						D[2] = castOp(t0_2);

						D[3] = castOp(t1_0);
						D[4] = castOp(t1_1);
						D[5] = castOp(t1_2);

						D[6] = castOp(t2_0);
						D[7] = castOp(t2_1);
						D[8] = castOp(t2_2);

						D[9] = castOp(t3_0);
						D[10] = castOp(t3_1);
						D[11] = castOp(t3_2);
#endif
						D += 12;
					}

					for (; dx < X1; dx++) 
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const short* w = wtab + FXY[dx] * 4;
						const uchar* S = S0 + sy *sstep + sx * 3;

						int t0 = S[0] * w[0] + S[3] * w[1] + S[sstep + 0] * w[2] + S[sstep + 3] * w[3];
						int t1 = S[1] * w[0] + S[4] * w[1] + S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
						int t2 = S[2] * w[0] + S[5] * w[1] + S[sstep + 2] * w[2] + S[sstep + 5] * w[3];

						D[0] = castOp_tq(t0); 
						D[1] = castOp_tq(t1);
						D[2] = castOp_tq(t2);

						D += 3;
					}
				} else if (cn == 4) {
					for (; dx < X1; dx++, D += 4) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const short* w = wtab + FXY[dx] * 4;
						const uchar* S = S0 + sy*sstep + sx * 4;
						int t0 = S[0] * w[0] + S[4] * w[1] + S[sstep] * w[2] + S[sstep + 4] * w[3];
						int t1 = S[1] * w[0] + S[5] * w[1] + S[sstep + 1] * w[2] + S[sstep + 5] * w[3];
						D[0] = castOp_tq(t0); D[1] = castOp_tq(t1);
						t0 = S[2] * w[0] + S[6] * w[1] + S[sstep + 2] * w[2] + S[sstep + 6] * w[3];
						t1 = S[3] * w[0] + S[7] * w[1] + S[sstep + 3] * w[2] + S[sstep + 7] * w[3];
						D[2] = castOp_tq(t0); D[3] = castOp_tq(t1);
					}
				} else {
					for (; dx < X1; dx++, D += cn) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const short* w = wtab + FXY[dx] * 4;
						const uchar* S = S0 + sy*sstep + sx*cn;
						for (k = 0; k < cn; k++) {
							int t0 = S[k] * w[0] + S[k + cn] * w[1] + S[sstep + k] * w[2] + S[sstep + k + cn] * w[3];
							D[k] = castOp_tq(t0);
						}
					}
				}
			} else {
				if (borderType == BORDER_TRANSPARENT && cn != 3) {
					D += (X1 - dx)*cn;
					dx = X1;
					continue;
				}

				if (cn == 1) {
					for (; dx < X1; dx++, D++) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
							D[0] = cval[0];
						} else {
							int sx0, sx1, sy0, sy1;
							uchar v0, v1, v2, v3;
							const short* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE) {
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0[sy0*sstep + sx0];
								v1 = S0[sy0*sstep + sx1];
								v2 = S0[sy1*sstep + sx0];
								v3 = S0[sy1*sstep + sx1];
							} else {
								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
							}
							D[0] = castOp_tq(int(v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]));
						}
					}
				} 
				else if (cn == 3 && borderType == BORDER_CONSTANT) {
					for (; dx < X1; dx++) {
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0) {
							D[0] = cval[0];
							D[1] = cval[1];
							D[2] = cval[2];
						} else {
							int sx0, sx1, sy0, sy1;
							const uchar *v0, *v1, *v2, *v3;
							const short* w = wtab + FXY[dx] * 4;

							sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
							sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
							sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
							sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
							v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
							v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
							v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
							v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
							
							D[0] = castOp_tq(int(v0[0] * w[0] + v1[0] * w[1] + v2[0] * w[2] + v3[0] * w[3]));
							D[1] = castOp_tq(int(v0[1] * w[0] + v1[1] * w[1] + v2[1] * w[2] + v3[1] * w[3]));
							D[2] = castOp_tq(int(v0[2] * w[0] + v1[2] * w[1] + v2[2] * w[2] + v3[2] * w[3]));
						}
						D += 3;
					}
				} else {
					for (; dx < X1; dx++, D += cn) 
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
							for (k = 0; k < cn; k++)
								D[k] = cval[k];
						} else {
							int sx0, sx1, sy0, sy1;
							const uchar *v0, *v1, *v2, *v3;
							const short* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE) {
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0 + sy0*sstep + sx0*cn;
								v1 = S0 + sy0*sstep + sx1*cn;
								v2 = S0 + sy1*sstep + sx0*cn;
								v3 = S0 + sy1*sstep + sx1*cn;
							} else if (borderType == BORDER_TRANSPARENT && ((unsigned)sx >= (unsigned)(ssize.width - 1) || (unsigned)sy >= (unsigned)(ssize.height - 1))) {
								continue;
							} else {
								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
							}
							for (k = 0; k < cn; k++)
								D[k] = castOp_tq(int(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3]));
						}
					}
				}
			}
		}
	}

	return 0;
}

template<class CastOp, typename AT, int ONE, typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remapBicubic(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst,
	const Mat_<_Tp2, chs2>& _xy, const Mat_<_Tp3, chs3>& _fxy, const void* _wtab, int borderType, const Scalar& _borderValue)
{
	typedef typename CastOp::rtype T;
	typedef typename CastOp::type1 WT;
	Size ssize = _src.size(), dsize = _dst.size();
	int cn = _src.channels;
	const AT* wtab = (const AT*)_wtab;
	const T* S0 = (const T*)_src.ptr();
	size_t sstep = _src.step / sizeof(S0[0]);
	Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
		saturate_cast<T>(_borderValue[1]),
		saturate_cast<T>(_borderValue[2]),
		saturate_cast<T>(_borderValue[3]));
	int dx, dy;
	CastOp castOp;
	int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

	unsigned width1 = std::max(ssize.width - 3, 0), height1 = std::max(ssize.height - 3, 0);

	for (dy = 0; dy < dsize.height; dy++) {
		T* D = (T*)_dst.ptr(dy);
		const short* XY = (const short*)_xy.ptr(dy);
		const ushort* FXY = (const ushort*)_fxy.ptr(dy);

		for (dx = 0; dx < dsize.width; dx++, D += cn) {
			int sx = XY[dx * 2] - 1, sy = XY[dx * 2 + 1] - 1;
			const AT* w = wtab + FXY[dx] * 16;
			int i, k;
			if ((unsigned)sx < width1 && (unsigned)sy < height1) {
				const T* S = S0 + sy*sstep + sx*cn;
				for (k = 0; k < cn; k++) {
					WT sum = S[0] * w[0] + S[cn] * w[1] + S[cn * 2] * w[2] + S[cn * 3] * w[3];
					S += sstep;
					sum += S[0] * w[4] + S[cn] * w[5] + S[cn * 2] * w[6] + S[cn * 3] * w[7];
					S += sstep;
					sum += S[0] * w[8] + S[cn] * w[9] + S[cn * 2] * w[10] + S[cn * 3] * w[11];
					S += sstep;
					sum += S[0] * w[12] + S[cn] * w[13] + S[cn * 2] * w[14] + S[cn * 3] * w[15];
					S += 1 - sstep * 3;
					D[k] = castOp(sum);
				}
			} else {
				int x[4], y[4];
				if (borderType == BORDER_TRANSPARENT &&
					((unsigned)(sx + 1) >= (unsigned)ssize.width ||
					(unsigned)(sy + 1) >= (unsigned)ssize.height))
					continue;

				if (borderType1 == BORDER_CONSTANT &&
					(sx >= ssize.width || sx + 4 <= 0 ||
					sy >= ssize.height || sy + 4 <= 0)) {
					for (k = 0; k < cn; k++)
						D[k] = cval[k];
					continue;
				}

				for (i = 0; i < 4; i++) {
					x[i] = borderInterpolate<int>(sx + i, ssize.width, borderType1)*cn;
					y[i] = borderInterpolate<int>(sy + i, ssize.height, borderType1);
				}

				for (k = 0; k < cn; k++, S0++, w -= 16) {
					WT cv = cval[k], sum = cv*ONE;
					for (i = 0; i < 4; i++, w += 4) {
						int yi = y[i];
						const T* S = S0 + yi*sstep;
						if (yi < 0)
							continue;
						if (x[0] >= 0)
							sum += (S[x[0]] - cv)*w[0];
						if (x[1] >= 0)
							sum += (S[x[1]] - cv)*w[1];
						if (x[2] >= 0)
							sum += (S[x[2]] - cv)*w[2];
						if (x[3] >= 0)
							sum += (S[x[3]] - cv)*w[3];
					}
					D[k] = castOp(sum);
				}
				S0 -= cn;
			}
		}
	}

	return 0;
}

template<class CastOp, typename AT, int ONE, typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remapLanczos4(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst,
	const Mat_<_Tp2, chs2>& _xy, const Mat_<_Tp3, chs3>& _fxy, const void* _wtab, int borderType, const Scalar& _borderValue)
{
	typedef typename CastOp::rtype T;
	typedef typename CastOp::type1 WT;
	Size ssize = _src.size(), dsize = _dst.size();
	int cn = _src.channels;
	const AT* wtab = (const AT*)_wtab;
	const T* S0 = (const T*)_src.ptr();
	size_t sstep = _src.step / sizeof(S0[0]);
	Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
		saturate_cast<T>(_borderValue[1]),
		saturate_cast<T>(_borderValue[2]),
		saturate_cast<T>(_borderValue[3]));
	int dx, dy;
	CastOp castOp;
	int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

	unsigned width1 = std::max(ssize.width - 7, 0), height1 = std::max(ssize.height - 7, 0);

	for (dy = 0; dy < dsize.height; dy++) {
		T* D = (T*)_dst.ptr(dy);
		const short* XY = (const short*)_xy.ptr(dy);
		const ushort* FXY = (const ushort*)_fxy.ptr(dy);

		for (dx = 0; dx < dsize.width; dx++, D += cn) {
			int sx = XY[dx * 2] - 3, sy = XY[dx * 2 + 1] - 3;
			const AT* w = wtab + FXY[dx] * 64;
			const T* S = S0 + sy*sstep + sx*cn;
			int i, k;
			if ((unsigned)sx < width1 && (unsigned)sy < height1) {
				for (k = 0; k < cn; k++) {
					WT sum = 0;
					for (int r = 0; r < 8; r++, S += sstep, w += 8)
						sum += S[0] * w[0] + S[cn] * w[1] + S[cn * 2] * w[2] + S[cn * 3] * w[3] +
						S[cn * 4] * w[4] + S[cn * 5] * w[5] + S[cn * 6] * w[6] + S[cn * 7] * w[7];
					w -= 64;
					S -= sstep * 8 - 1;
					D[k] = castOp(sum);
				}
			} else {
				int x[8], y[8];
				if (borderType == BORDER_TRANSPARENT &&
					((unsigned)(sx + 3) >= (unsigned)ssize.width ||
					(unsigned)(sy + 3) >= (unsigned)ssize.height))
					continue;

				if (borderType1 == BORDER_CONSTANT &&
					(sx >= ssize.width || sx + 8 <= 0 ||
					sy >= ssize.height || sy + 8 <= 0)) {
					for (k = 0; k < cn; k++)
						D[k] = cval[k];
					continue;
				}

				for (i = 0; i < 8; i++) {
					x[i] = borderInterpolate<int>(sx + i, ssize.width, borderType1)*cn;
					y[i] = borderInterpolate<int>(sy + i, ssize.height, borderType1);
				}

				for (k = 0; k < cn; k++, S0++, w -= 64) {
					WT cv = cval[k], sum = cv*ONE;
					for (i = 0; i < 8; i++, w += 8) {
						int yi = y[i];
						const T* S1 = S0 + yi*sstep;
						if (yi < 0)
							continue;
						if (x[0] >= 0)
							sum += (S1[x[0]] - cv)*w[0];
						if (x[1] >= 0)
							sum += (S1[x[1]] - cv)*w[1];
						if (x[2] >= 0)
							sum += (S1[x[2]] - cv)*w[2];
						if (x[3] >= 0)
							sum += (S1[x[3]] - cv)*w[3];
						if (x[4] >= 0)
							sum += (S1[x[4]] - cv)*w[4];
						if (x[5] >= 0)
							sum += (S1[x[5]] - cv)*w[5];
						if (x[6] >= 0)
							sum += (S1[x[6]] - cv)*w[6];
						if (x[7] >= 0)
							sum += (S1[x[7]] - cv)*w[7];
					}
					D[k] = castOp(sum);
				}
				S0 -= cn;
			}
		}
	}

	return 0;
}

template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remap_nearest(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue)
{
	const void* ctab = 0;
	bool fixpt = typeid(uchar).name() == typeid(_Tp1).name();
	bool planar_input = map1.channels == 1;
	Range range(0, dst.rows);

	int x, y, x1, y1;
	const int buf_size = 1 << 14;
	int brows0 = std::min(128, dst.rows);
	int bcols0 = std::min(buf_size / brows0, dst.cols);
	brows0 = std::min(buf_size / bcols0, dst.rows);

	Mat_<short, 2> _bufxy(brows0, bcols0);
	Mat_<short, 2> map1_tmp1(map1.rows, map1.cols, map1.data);
	Mat_<float, 2> map1_tmp2(map1.rows, map1.cols, map1.data);

	for (y = range.start; y < range.end; y += brows0) {
		for (x = 0; x < dst.cols; x += bcols0) {
			int brows = std::min(brows0, range.end - y);
			int bcols = std::min(bcols0, dst.cols - x);
			Mat_<_Tp1, chs1> dpart;
			dst.getROI(dpart, Rect(x, y, bcols, brows));
			Mat_<short, 2> bufxy;
			_bufxy.getROI(bufxy, Rect(0, 0, bcols, brows));

			if (map1.channels == 2 && sizeof(_Tp2) == sizeof(short) && map2.empty()) { // the data is already in the right format
				map1_tmp1.getROI(bufxy, Rect(x, y, bcols, brows));
			} else if (sizeof(_Tp2) != sizeof(float)) {
				for (y1 = 0; y1 < brows; y1++) {
					short* XY = (short*)bufxy.ptr(y1);
					const short* sXY = (const short*)map1.ptr(y + y1) + x * 2;
					const ushort* sA = (const ushort*)map2.ptr(y + y1) + x;

					for (x1 = 0; x1 < bcols; x1++) {
						int a = sA[x1] & (INTER_TAB_SIZE2 - 1);
						XY[x1 * 2] = sXY[x1 * 2] + NNDeltaTab_i[a][0];
						XY[x1 * 2 + 1] = sXY[x1 * 2 + 1] + NNDeltaTab_i[a][1];
					}
				}
			} else if (!planar_input) {
				map1_tmp2.convertTo(bufxy);
			} else {
				for (y1 = 0; y1 < brows; y1++) {
					short* XY = (short*)bufxy.ptr(y1);
					const float* sX = (const float*)map1.ptr(y + y1) + x;
					const float* sY = (const float*)map2.ptr(y + y1) + x;

					x1 = 0;
					for (; x1 < bcols; x1++) {
						XY[x1 * 2] = saturate_cast<short>(sX[x1]);
						XY[x1 * 2 + 1] = saturate_cast<short>(sY[x1]);
					}
				}
			}

			remapNearest<_Tp1, short, chs1, 2>(src, dpart, bufxy, borderMode, borderValue);
		}
	}

	return 0;
}

#if __ARM_NEON
inline int32x4_t v_round(const float32x4_t& a)
{
    static const int32x4_t v_sign = vdupq_n_s32(1 << 31),
        v_05 = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));

    int32x4_t v_addition = vorrq_s32(v_05, vandq_s32(v_sign, vreinterpretq_s32_f32(a)));
    return int32x4_t(vcvtq_s32_f32(vaddq_f32(a, vreinterpretq_f32_s32(v_addition))));
}
#endif

template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remap_linear(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue)
{
	const void* ctab = 0;
	bool fixpt = typeid(uchar).name() == typeid(_Tp1).name();
	bool planar_input = map1.channels == 1;
	ctab = initInterTab2D<_Tp1>(INTER_LINEAR, fixpt);
	Range range(0, dst.rows);

	int x, y, x1, y1;
	const int buf_size = 1 << 14;
	int brows0 = std::min(128, dst.rows);
	int bcols0 = std::min(buf_size / brows0, dst.cols);
	brows0 = std::min(buf_size / bcols0, dst.rows);

	Mat_<short, 2> _bufxy(brows0, bcols0);
	Mat_<ushort, 1> _bufa(brows0, bcols0);
	Mat_<short, 2> map1_tmp1(map1.rows, map1.cols, map1.data);

	for (y = range.start; y < range.end; y += brows0) {
		for (x = 0; x < dst.cols; x += bcols0) {
			int brows = std::min(brows0, range.end - y);
			int bcols = std::min(bcols0, dst.cols - x);
			Mat_<_Tp1, chs1> dpart;
			dst.getROI(dpart, Rect(x, y, bcols, brows));
			Mat_<short, 2> bufxy;
			_bufxy.getROI(bufxy, Rect(0, 0, bcols, brows));
			Mat_<ushort, 1> bufa;
			_bufa.getROI(bufa, Rect(0, 0, bcols, brows));

			for (y1 = 0; y1 < brows; y1++) {
				short* XY = (short*)bufxy.ptr(y1);
				ushort* A = (ushort*)bufa.ptr(y1);

				if (map1.channels == 2 && typeid(short).name() == typeid(_Tp2).name() &&
					(map2.channels == 1 && sizeof(_Tp3) == 2)) {
					map1_tmp1.getROI(bufxy, Rect(x, y, bcols, brows));

					const ushort* sA = (const ushort*)map2.ptr(y + y1) + x;
					x1 = 0;
#if __ARM_NEON
					uint16x8_t v_scale = vdupq_n_u16(INTER_TAB_SIZE2 - 1);
					int span = 8;
					for( ; x1 <= bcols - span; x1 += span )
						vst1q_u16((unsigned short*)(A + x1), vld1q_u16(sA + x1) & v_scale);
#endif // __ARM_NEON
					for (; x1 < bcols; x1++)
						A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
				} else if (planar_input) {
					const float* sX = (const float*)map1.ptr(y + y1) + x;
					const float* sY = (const float*)map2.ptr(y + y1) + x;

					x1 = 0;
#if __ARM_NEON
					float32x4_t v_scale = vdupq_n_f32((float)INTER_TAB_SIZE);
					int32x4_t v_scale2 = vdupq_n_s32(INTER_TAB_SIZE - 1);
					int span = 4;
					for( ; x1 <= bcols - span * 2; x1 += span * 2 )
					{
						int32x4_t v_sx0 = v_round(v_scale * vld1q_f32(sX + x1));
						int32x4_t v_sy0 = v_round(v_scale * vld1q_f32(sY + x1));
						int32x4_t v_sx1 = v_round(v_scale * vld1q_f32(sX + x1 + span));
						int32x4_t v_sy1 = v_round(v_scale * vld1q_f32(sY + x1 + span));
						uint16x8_t v_sx8 = vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(v_sx0 & v_scale2), vmovn_s32(v_sx1 & v_scale2)));
						uint16x8_t v_sy8 = vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(v_sy0 & v_scale2), vmovn_s32(v_sy1 & v_scale2)));
						uint16x8_t v_v = vshlq_n_u16(v_sy8, INTER_BITS) | (v_sx8);
						vst1q_u16(A + x1, v_v);

						int16x8_t v_d0 = vcombine_s16(vmovn_s32(vshrq_n_s32(v_sx0, INTER_BITS)), vmovn_s32(vshrq_n_s32(v_sx1, INTER_BITS)));
						int16x8_t v_d1 = vcombine_s16(vmovn_s32(vshrq_n_s32(v_sy0, INTER_BITS)), vmovn_s32(vshrq_n_s32(v_sy1, INTER_BITS)));
						int16x8x2_t v_tmp;
						v_tmp.val[0] = v_d0;
						v_tmp.val[1] = v_d1;
						vst2q_s16(XY + (x1 << 1), v_tmp);
					}
#endif // __ARM_NEON					
					for (; x1 < bcols; x1++) {
						int sx = fbcRound(sX[x1] * INTER_TAB_SIZE);
						int sy = fbcRound(sY[x1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				} else {
					const float* sXY = (const float*)map1.ptr(y + y1) + x * 2;
					x1 = 0;
#if __ARM_NEON
					float32x4_t v_scale = vdupq_n_f32((float)INTER_TAB_SIZE);
					int32x4_t v_scale2 = vdupq_n_s32(INTER_TAB_SIZE - 1), v_scale3 = vdupq_n_s32(INTER_TAB_SIZE);
					int span = 4;
					for( ; x1 <= bcols - span * 2; x1 += span * 2 )
					{
						float32x4_t v_fx, v_fy;
						float32x4x2_t v_tmp_f32 = vld2q_f32(sXY + (x1 << 1));
						v_fx = v_tmp_f32.val[0];
						v_fy = v_tmp_f32.val[1];

						int32x4_t v_sx0 = v_round(v_fx * v_scale);
						int32x4_t v_sy0 = v_round(v_fy * v_scale);
						v_tmp_f32 = vld2q_f32(sXY + ((x1 + span) << 1));
						v_fx = v_tmp_f32.val[0];
						v_fy = v_tmp_f32.val[1];						

						int32x4_t v_sx1 = v_round(v_fx * v_scale);
						int32x4_t v_sy1 = v_round(v_fy * v_scale);
						int32x4_t v_v0 = vmlaq_s32(v_scale3, (v_sy0 & v_scale2), (v_sx0 & v_scale2));
						int32x4_t v_v1 = vmlaq_s32(v_scale3, (v_sy1 & v_scale2), (v_sx1 & v_scale2));
						uint16x8_t v_v8 = vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(v_v0), vmovn_s32(v_v1)));
						vst1q_u16(A + x1, v_v8);

						int16x8_t v_dx = vcombine_s16(vmovn_s32(vshrq_n_s32(v_sx0, INTER_BITS)), vmovn_s32(vshrq_n_s32(v_sx1, INTER_BITS)));
						int16x8_t v_dy = vcombine_s16(vmovn_s32(vshrq_n_s32(v_sy0, INTER_BITS)), vmovn_s32(vshrq_n_s32(v_sy1, INTER_BITS)));
						int16x8x2_t v_tmp_s16;
						v_tmp_s16.val[0] = v_dx;
						v_tmp_s16.val[1] = v_dy;
						vst2q_s16(XY + (x1 << 1), v_tmp_s16);						
					}
#endif // __ARM_NEON					
					for (x1 = 0; x1 < bcols; x1++) {
						int sx = fbcRound(sXY[x1 * 2] * INTER_TAB_SIZE);
						int sy = fbcRound(sXY[x1 * 2 + 1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				}
			}

			if (typeid(_Tp1).name() == typeid(uchar).name()) { // uchar
				remapBilinear_fixed<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			} else { // float
				remapBilinear<Cast<float, float>, float, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			}	
		}
	}

	return 0;
}

template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remap_cubic(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue)
{
	const void* ctab = 0;
	bool fixpt = typeid(uchar).name() == typeid(_Tp1).name();
	bool planar_input = map1.channels == 1;
	ctab = initInterTab2D<_Tp1>(INTER_CUBIC, fixpt);
	Range range(0, dst.rows);

	int x, y, x1, y1;
	const int buf_size = 1 << 14;
	int brows0 = std::min(128, dst.rows);
	int bcols0 = std::min(buf_size / brows0, dst.cols);
	brows0 = std::min(buf_size / bcols0, dst.rows);

	Mat_<short, 2> _bufxy(brows0, bcols0);
	Mat_<ushort, 1> _bufa(brows0, bcols0);
	Mat_<short, 2> map1_tmp1(map1.rows, map1.cols, map1.data);

	for (y = range.start; y < range.end; y += brows0) {
		for (x = 0; x < dst.cols; x += bcols0) {
			int brows = std::min(brows0, range.end - y);
			int bcols = std::min(bcols0, dst.cols - x);
			Mat_<_Tp1, chs1> dpart;
			dst.getROI(dpart, Rect(x, y, bcols, brows));
			Mat_<short, 2> bufxy;
			_bufxy.getROI(bufxy, Rect(0, 0, bcols, brows));
			Mat_<ushort, 1> bufa;
			_bufa.getROI(bufa, Rect(0, 0, bcols, brows));

			for (y1 = 0; y1 < brows; y1++) {
				short* XY = (short*)bufxy.ptr(y1);
				ushort* A = (ushort*)bufa.ptr(y1);

				if (map1.channels == 2 && typeid(short).name() == typeid(_Tp2).name() &&
					(map2.channels == 1 && sizeof(_Tp3) == 2)) {
					map1_tmp1.getROI(bufxy, Rect(x, y, bcols, brows));

					const ushort* sA = (const ushort*)map2.ptr(y + y1) + x;
					x1 = 0;

					for (; x1 < bcols; x1++)
						A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
				} else if (planar_input) {
					const float* sX = (const float*)map1.ptr(y + y1) + x;
					const float* sY = (const float*)map2.ptr(y + y1) + x;

					x1 = 0;
					for (; x1 < bcols; x1++) {
						int sx = fbcRound(sX[x1] * INTER_TAB_SIZE);
						int sy = fbcRound(sY[x1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				} else {
					const float* sXY = (const float*)map1.ptr(y + y1) + x * 2;
					x1 = 0;
					for (x1 = 0; x1 < bcols; x1++) {
						int sx = fbcRound(sXY[x1 * 2] * INTER_TAB_SIZE);
						int sy = fbcRound(sXY[x1 * 2 + 1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				}
			}

			if (typeid(_Tp1).name() == typeid(uchar).name()) { // uchar
				remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			} else { // float
				remapBicubic<Cast<float, float>, float, 1, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			}
		}
	}

	return 0;
}

template<typename _Tp1, typename _Tp2, typename _Tp3, int chs1, int chs2, int chs3>
static int remap_lanczos4(const Mat_<_Tp1, chs1>& src, Mat_<_Tp1, chs1>& dst,
	const Mat_<_Tp2, chs2>& map1, const Mat_<_Tp3, chs3>& map2, int borderMode, const Scalar& borderValue)
{
	const void* ctab = 0;
	bool fixpt = typeid(uchar).name() == typeid(_Tp1).name();
	bool planar_input = map1.channels == 1;
	ctab = initInterTab2D<_Tp1>(INTER_LANCZOS4, fixpt);
	Range range(0, dst.rows);

	int x, y, x1, y1;
	const int buf_size = 1 << 14;
	int brows0 = std::min(128, dst.rows);
	int bcols0 = std::min(buf_size / brows0, dst.cols);
	brows0 = std::min(buf_size / bcols0, dst.rows);

	Mat_<short, 2> _bufxy(brows0, bcols0);
	Mat_<ushort, 1> _bufa(brows0, bcols0);
	Mat_<short, 2> map1_tmp1(map1.rows, map1.cols, map1.data);

	for (y = range.start; y < range.end; y += brows0) {
		for (x = 0; x < dst.cols; x += bcols0) {
			int brows = std::min(brows0, range.end - y);
			int bcols = std::min(bcols0, dst.cols - x);
			Mat_<_Tp1, chs1> dpart;
			dst.getROI(dpart, Rect(x, y, bcols, brows));
			Mat_<short, 2> bufxy;
			_bufxy.getROI(bufxy, Rect(0, 0, bcols, brows));
			Mat_<ushort, 1> bufa;
			_bufa.getROI(bufa, Rect(0, 0, bcols, brows));

			for (y1 = 0; y1 < brows; y1++) {
				short* XY = (short*)bufxy.ptr(y1);
				ushort* A = (ushort*)bufa.ptr(y1);

				if (map1.channels == 2 && typeid(short).name() == typeid(_Tp2).name() &&
					(map2.channels == 1 && sizeof(_Tp3) == 2)) {
					map1_tmp1.getROI(bufxy, Rect(x, y, bcols, brows));

					const ushort* sA = (const ushort*)map2.ptr(y + y1) + x;
					x1 = 0;

					for (; x1 < bcols; x1++)
						A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
				} else if (planar_input) {
					const float* sX = (const float*)map1.ptr(y + y1) + x;
					const float* sY = (const float*)map2.ptr(y + y1) + x;

					x1 = 0;
					for (; x1 < bcols; x1++) {
						int sx = fbcRound(sX[x1] * INTER_TAB_SIZE);
						int sy = fbcRound(sY[x1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				} else {
					const float* sXY = (const float*)map1.ptr(y + y1) + x * 2;
					x1 = 0;
					for (x1 = 0; x1 < bcols; x1++) {
						int sx = fbcRound(sXY[x1 * 2] * INTER_TAB_SIZE);
						int sy = fbcRound(sXY[x1 * 2 + 1] * INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				}
			}

			if (typeid(_Tp1).name() == typeid(uchar).name()) { // uchar
				remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			}
			else { // float
				remapLanczos4<Cast<float, float>, float, 1, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
			}
		}
	}

	return 0;
}

} // namespace fbc

#endif // FBC_CV_REMAP_HPP_
