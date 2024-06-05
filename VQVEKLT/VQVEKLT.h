#pragma once
#include "stb_image.h"
#include "stb_image_write.h"
#include "MatrixCalculator.h"
#include <iostream>
#include <bitset>
#include <limits>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <random>
#include <Eigen/Dense>
#define DCTV4_SIZE 16
#define DCTV8_SIZE 64
#define DCTV16_SIZE 256
#define DCTV32_SIZE 1024
#define EVmatrixes_ROWS 64
#define EVmatrixes_COLS 253952
#define R_ROWS 64
#define R_COLS 4096
#define R4_ROWS 16
#define R4_COLS 11
#define EV4_ROWS 16
#define EV4_COLS 154
#define Z4_SIZE 15
#define Z8_SIZE 63
#define Z16_SIZE 255
#define Z32_SIZE 1023
#define JHuffTable_ROWS 162
#define JHuffTable_COLS 20

#define SAVE_EXTRA_FILE false
#define ELAPSED_TIME false
#define ELAPSED_TIME_TOTAL true
#define COLOR_MODE true

using namespace Eigen;

class VQVEKLT
{
public:
	VQVEKLT();
	~VQVEKLT();

	std::string folderPath;
	std::vector<std::string> imagePath;
	void listImagePath(const std::string& directory);
	void writeGrayScaleImage(const std::string& filePath);
	void loadInputImg(const std::string& fname);
	void convertToGrayScale();
	void convertToYCbCr();
	void enCode(const std::string& fname, float q, int eob_thresh_y, int eob_thresh_cbcr);
	void enCodeYCbCr(float q, int mode); // mode : 0 (Y), 1 (Cb), 2 (Cr)
	void deCode(const std::string& fname);
	void deCodeYCbCr(const char* buffer, const VectorXi& header, int mode, int& start, int& end); // mode : 0 (Y), 1 (Cb), 2 (Cr)
	
	std::string extractFilename(const std::string& path);
	std::string changeExtension(const std::string& filename, const std::string& newExtension);
	void afterAnalysis(const std::string& decoded_fname);
	void floatImageDataToRGB(MatrixXf& _R, MatrixXf& _G, MatrixXf& _B);
	void genRandoms(int total, int num_to_pick, std::vector<int>& pickedNumbers);

private:
	template <typename T>
	void loadBinFile(const std::string& fname, T& arr, int mode); // mode 0 : float, mode 1 : int
	template <typename T>
	T im2col(const T& image, int blockSize);
	template <typename T>
	void col2im(T& cols, T& im, int blockWidth, int blockHeight);
	void findEOB(const MatrixXf& Qac, VectorXi& eobN, VectorXi& aco);
	void insertEOB(const MatrixXf& aci, VectorXi& aco);
	int getMinQdcFlag(int min, bool exist);

	void matrixCal_By_ARow(const MatrixXf& A, const MatrixXf& B, MatrixXf& C);
	void matrixCal_By_BCol(const MatrixXf& A, const MatrixXf& B, MatrixXf& C);

	void onlyNonZeroElements(const VectorXi& seq, VectorXi& cdf, VectorXi& seq_ind);
	void getCdfAndSeqIdx(const VectorXi& seq, VectorXi& cdf, VectorXi& seq_ind, int& min, int& max);
	void arithmeticEncode(const VectorXi& seq, std::vector<char>& code);
	int arithmeticDecode(const std::vector<char>& code, VectorXi& seq);
	void _arithmeticEncode(const VectorXi& cdf, const VectorXi& seq_ind, VectorXi& binaryArray);
	void _arithmeticDecode(const VectorXi& cdf, const VectorXi& binaryArray, VectorXi& seq_ind);
	void indN2ari(const VectorXi& seq, int indN_size, std::vector<char>& code);
	void ari2indN(const std::vector<char>& code, int indN_size, VectorXi& seq);
	void signAC2ari(const VectorXi& seq, int signAC_size, std::vector<char>& code);
	void ari2signAC(const std::vector<char>& code, VectorXi& seq);
	
	void golombEncode(const VectorXi& seq, VectorXi& binaryArray);
	int golombDecode(const VectorXi& binaryArray, VectorXi& seq);
	
	void qtlevel3(VectorXi& ind8, int width8, int height8, VectorXi& qtind8, VectorXi& qtind16, VectorXi& qtind32);
	void getSelected(const MatrixXf& input, const VectorXi& ind, const int val, MatrixXf& selected);
	void prepareCodingForVEKLT(const MatrixXf& y, VectorXi& out_Sign, VectorXi& out_AC);
	void prepareCodingForDCT(const MatrixXf& y, const int zSize, VectorXi& out_Sign, VectorXi& out_AC, VectorXi& out_DC);
	
	void dc2ari(const VectorXi& qdc, std::vector<char>& code, int& min);
	void vq2ari(const VectorXi& qvq, std::vector<char>& code, int& min);
	void ari2dc(const std::vector<char>& code, int min, VectorXi& qdc);
	void ari2vq(const std::vector<char>& code, int min, VectorXi& qvq);
	void decodeDCcoefficient(const char* buffer, const int min_qdc, int& start, int& end, int length, VectorXi& qdc);

	void rind2code(const VectorXi& Rind, std::vector<char>& code);
	void code2rind(const std::vector<char>& charArray, VectorXi& Rind);
	void decodeRindAndVQcoefficient(const char* buffer, int mode, int min_qvq, int& start, int& end, int rind_length, int qvq_length, VectorXi& Rind, VectorXi& qvq);
	void decodeHeader(const char* buffer, int mode, int& start, int& end, VectorXi& header); // mode : 0 (Y), 1 (Cb), 2 (Cr)

	void convertBinaryToChar(const VectorXi& binaryArray, std::vector<char>& charArray);
	void convertCharToBinary(const std::vector<char>& charArray, VectorXi& binaryArray);
	
	void decodeBinaryIndexImage(const char* buffer, int indN_size, int& start, int& end, int length, VectorXi& indN);
	void decodeSignACcoefficient(const char* buffer, int& start, int& end, int length, VectorXi& signQac);
	int decodeAbsACcoefficient(const char* buffer, int& start, int& end, int length, VectorXi& absQac);
	void colMreshape(const VectorXi& abs_seq, const VectorXi& sign_seq, int eobsym, MatrixXf& colMat);
	
	void appendRowToCSV(const std::string& filename, const std::vector<std::string>& row, bool append);
	std::string getFileNameWithExtension(const std::string& filename);
	void saveDecodedImage(const MatrixXf& Rim, int mode, int padN_h, int padN_w, int N);
	void saveY2BMP(const MatrixXf& matrix, const std::string& filename);
	void saveRGB2BMP(const MatrixXf& R, const MatrixXf& G, const MatrixXf& B, const std::string& filename);
	void scaleUpMatrix(const MatrixXf& input, MatrixXf& output, int output_width, int output_height);
	void YCbCr2RGB(const MatrixXf& Y, const MatrixXf& Cb, const MatrixXf& Cr, MatrixXf& R, MatrixXf& G, MatrixXf& B);
	float limitVal(float val, float min, float max);

	void writeVectorXi(const VectorXi& vector, const std::string& filename);
	void writeMatrixXf(const MatrixXf& matrix, const std::string& filename);
	float calculatePSNR(const MatrixXf& img1, const MatrixXf& img2, bool getMSE = false);
	float calculateCPSNR(const MatrixXf& img1_R, const MatrixXf& img1_G, const MatrixXf& img1_B, const MatrixXf& img2_R, const MatrixXf& img2_G, const MatrixXf& img2_B, bool getMSE = false);

	MatrixXf DCTV4; std::string DCTV4_fname = "DCTV4";
	MatrixXf DCTV8; std::string DCTV8_fname = "DCTV8";
	MatrixXf DCTV16; std::string DCTV16_fname = "DCTV16";
	MatrixXf DCTV32; std::string DCTV32_fname = "DCTV32";
	
	MatrixXf EVmatrixes; std::string EVmatrixes_fname = "EVmatrixes4096";
	MatrixXf R; std::string R_fname = "R4096";

	MatrixXf R4; std::string R4_fname = "R4_CbCr";
	MatrixXf EV_Cb; std::string EV_Cb_fname = "EV_Cb";
	MatrixXf EV_Cr; std::string EV_Cr_fname = "EV_Cr";
	
	VectorXi Z4;
	VectorXi Z8; std::string Z8_fname = "Z8";
	VectorXi Z16; std::string Z16_fname = "Z16";
	VectorXi Z32; std::string Z32_fname = "Z32";

	MatrixXf Y_bak;
	MatrixXf Cb_bak;

	std::string inputImgName;
	std::string inputVjpName;
	std::string binFolderName = "bin";
	std::string encodeFolderName = "encoded";
	std::string decodeFolderName = "decoded";
	std::string analysisFolderName = "analysis";
	std::string csvFileName = "result.csv";
	std::vector<std::string> csv_header;
	int cores;
	int num_threads;
	std::streampos fileSize;
	
	int tmpeobsym = 9999;
	int width;
	int width2;
	int widthN;
	int width_bak;
	int height;
	int height2;
	int heightN;
	int height_bak;
	int numChannels;
	unsigned char* imageData = nullptr;
	float* floatImageData = nullptr;
	float* floatImageGrayData = nullptr;
	float* floatImageCb = nullptr;
	float* floatImageCr = nullptr;
	float _q;

	int encode_time; // ms
	int decode_time; // ms
	VectorXf veklt_rate;
	VectorXf dctN_rate;
	VectorXf dctN2_rate;
	VectorXf dctN4_rate;

	int _eob_thresh_y;
	int _eob_thresh_cbcr;
};

