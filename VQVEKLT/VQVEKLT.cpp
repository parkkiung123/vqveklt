#include "VQVEKLT.h"

VQVEKLT::VQVEKLT()
{
    DCTV8 = MatrixXf(DCTV8_SIZE, DCTV8_SIZE);
    loadBinFile(DCTV8_fname, DCTV8, 0);
    DCTV16 = MatrixXf(DCTV16_SIZE, DCTV16_SIZE);
    loadBinFile(DCTV16_fname, DCTV16, 0);
    DCTV32 = MatrixXf(DCTV32_SIZE, DCTV32_SIZE);
    loadBinFile(DCTV32_fname, DCTV32, 0);
    EVmatrixes = MatrixXf(EVmatrixes_ROWS, EVmatrixes_COLS);
    loadBinFile(EVmatrixes_fname, EVmatrixes, 0);
    R = MatrixXf(R_ROWS, R_COLS);
    loadBinFile(R_fname, R, 0);
    
#if COLOR_MODE
    DCTV4 = MatrixXf(DCTV4_SIZE, DCTV4_SIZE);
    loadBinFile(DCTV4_fname, DCTV4, 0);
    R4 = MatrixXf(R4_ROWS, R4_COLS);
    loadBinFile(R4_fname, R4, 0);
    EV_Cb = MatrixXf(EV4_ROWS, EV4_COLS);
    loadBinFile(EV_Cb_fname, EV_Cb, 0);
    EV_Cr = MatrixXf(EV4_ROWS, EV4_COLS);
    loadBinFile(EV_Cr_fname, EV_Cr, 0);
    Z4 = VectorXi(Z4_SIZE);
    Z4 << 1, 2, 8, 0, 3, 7, 9, 4, 6 ,10, 13, 5, 11, 12, 14;
#endif

    Z8 = VectorXi(Z8_SIZE);
    loadBinFile(Z8_fname, Z8, 1);
    Z16 = VectorXi(Z16_SIZE);
    loadBinFile(Z16_fname, Z16, 1);
    Z32 = VectorXi(Z32_SIZE);
    loadBinFile(Z32_fname, Z32, 1);

    veklt_rate = -1 * VectorXf::Ones(3);
    dctN_rate = -1 * VectorXf::Ones(3);
    dctN2_rate = -1 * VectorXf::Ones(3);
    dctN4_rate = -1 * VectorXf::Ones(3);

    csv_header = {"fileName","size(WxH)","numChannels","q","bpp","psnr","EOB_THRESHOLD_Y","EOB_THRESHOLD_CBCR",
        "veklt_rate_Y","veklt_rate_Cb","veklt_rate_Cr",
        "dctN_rate_Y","dctN_rate_Cb","dctN_rate_Cr",
        "dctN2_rate_Y","dctN2_rate_Cb","dctN2_rate_Cr",
        "dctN4_rate_Y","dctN4_rate_Cb","dctN4_rate_Cr",
        "encode_time(ms)","decode_time(ms)"};
    std::string csv_path = analysisFolderName + "\\" + csvFileName;
    {
        std::ifstream file(csv_path);
        appendRowToCSV(csv_path, csv_header, false);
    }

    cores = std::thread::hardware_concurrency();
    num_threads = cores - 1;
}

VQVEKLT::~VQVEKLT()
{
    if (imageData != nullptr) free(imageData);
    if (floatImageData != nullptr) free(floatImageData);
    if (floatImageGrayData != nullptr) free(floatImageGrayData);
    if (floatImageCb != nullptr) free(floatImageCb);
    if (floatImageCr != nullptr) free(floatImageCr);
}

template <typename T>
void VQVEKLT::loadBinFile(const std::string& fname, T& mat, int mode)
{
    std::string path = binFolderName + "\\" + fname + ".bin";
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        if (mode == 0) {
            file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(float));
        }
        else if (mode == 1) {
            file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(int));
        }
        file.close();
    }
    else {
        std::cerr << "Error opening file for reading : "  + fname << std::endl;
        return;
    }
}

void VQVEKLT::listImagePath(const std::string& directory)
{
    folderPath = directory;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        auto ext = entry.path().extension();
        if (entry.is_regular_file() && (ext == ".jpg" || ext == ".png" || ext == ".bmp")) {
            imagePath.push_back(entry.path().string());
        }
    }
}

void VQVEKLT::writeGrayScaleImage(const std::string& filePath)
{
    std::ofstream outFile(filePath, std::ios::out | std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(floatImageGrayData), width * height * sizeof(float));
        outFile.close();
        std::cout << filePath + " successfully written to binary file." << std::endl;
    }
    else {
        std::cerr << "Error: Unable to open file for writing : " + filePath << std::endl;
    }
}

void VQVEKLT::loadInputImg(const std::string& fname)
{
    inputImgName = extractFilename(fname);

    imageData = stbi_load(fname.c_str(), &width, &height, &numChannels, 0);
    if (!imageData) {
        std::cerr << "Failed to load image: " << fname << std::endl;
        exit(1);
    }

    floatImageData = (float*)malloc(width * height * numChannels * sizeof(float));
    if (floatImageData != nullptr) {
        for (int i = 0; i < width * height * numChannels; i++) {
            floatImageData[i] = static_cast<float>(imageData[i]);
        }
    }
}

void VQVEKLT::saveY2BMP(const MatrixXf& matrix, const std::string& filename)
{
    int height = matrix.rows();
    int width = matrix.cols();

    std::vector<unsigned char> data(width * height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float val = limitVal(matrix(i, j) + 128.0f, 0.0f, 255.0f);
            data[i * width + j] = static_cast<unsigned char>(val);
        }
    }

    std::string path = decodeFolderName + "\\" + std::string(filename);
    stbi_write_bmp(path.c_str(), width, height, 1, data.data());
}

void VQVEKLT::saveRGB2BMP(const MatrixXf& R, const MatrixXf& G, const MatrixXf& B, const std::string& filename)
{
    int rows = R.rows();
    int cols = R.cols();

    std::vector<uint8_t> imageData(rows * cols * 3);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            imageData[3 * (i * cols + j) + 0] = static_cast<unsigned char>(limitVal(R(i, j), 0.0f, 255.0f));
            imageData[3 * (i * cols + j) + 1] = static_cast<unsigned char>(limitVal(G(i, j), 0.0f, 255.0f));
            imageData[3 * (i * cols + j) + 2] = static_cast<unsigned char>(limitVal(B(i, j), 0.0f, 255.0f));
        }
    }

    std::string path = decodeFolderName + "\\" + std::string(filename);
    stbi_write_bmp(path.c_str(), cols, rows, 3, imageData.data());
}

void VQVEKLT::scaleUpMatrix(const MatrixXf& input, MatrixXf& output, int output_width, int output_height)
{
    int rows = input.rows();
    int cols = input.cols();

    output = MatrixXf::Zero(output_height, output_width);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = input(i, j);
            output(2 * i, 2 * j) = val;
            output(2 * i + 1, 2 * j) = val;
            output(2 * i, 2 * j + 1) = val;
            output(2 * i + 1, 2 * j + 1) = val;
        }
    }

    if (output_height % 2 == 1) {
        int cols2 = cols * 2;
        int finalRowIdx = output_height - 1;
        for (int j = 0; j < cols2; j++) {
            output(finalRowIdx, j) = output(finalRowIdx - 1, j);
        }
    }
    if (output_width % 2 == 1) {
        int rows2 = rows * 2;
        int finalColIdx = output_width - 1;
        for (int i = 0; i < rows2; i++) {
            output(i, finalColIdx) = output(i, finalColIdx - 1);
        }
    }
    if (output_height % 2 == 1 && output_width % 2 == 1) {
        output(output_height - 1, output_width - 1) = output(output_height - 2, output_width - 2);
    }    
}

void VQVEKLT::YCbCr2RGB(const MatrixXf& Y, const MatrixXf& Cb, const MatrixXf& Cr, MatrixXf& R, MatrixXf& G, MatrixXf& B)
{
    int rows = Y.rows();
    int cols = Y.cols();
    
    R.resize(rows, cols);
    G.resize(rows, cols);
    B.resize(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float y = limitVal(Y(i, j) + 128.0f, 0.0f, 255.0f);
            float cb = limitVal(Cb(i, j), -128.0f, 127.0f);
            float cr = limitVal(Cr(i, j), -128.0f, 127.0f);
            R(i, j) = round(y + 1.402f * cr);
            G(i, j) = round(y - 0.344136f * cb - 0.714136f * cr);
            B(i, j) = round(y + 1.772f * cb);
        }
    }
}

float VQVEKLT::limitVal(float val, float min, float max)
{
    if (val < min) return min;
    else if (val > max) return max;
    else return val;
}

void VQVEKLT::writeVectorXi(const VectorXi& vector, const std::string& filename)
{
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        int rows = 1;
        int cols = vector.size();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        file.write(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(float));
        file.close();
    }
    else {
        std::cerr << "Unable to open file." << std::endl;
        return;
    }
}

void VQVEKLT::writeMatrixXf(const MatrixXf& matrix, const std::string& filename)
{
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        file.write(reinterpret_cast<const char*>(matrix.data()), matrix.rows() * matrix.cols() * sizeof(float));
        file.close();
    }
    else {
        std::cerr << "Unable to open file." << std::endl;
        return;
    }
}

float VQVEKLT::calculatePSNR(const MatrixXf& img1, const MatrixXf& img2, bool getMSE)
{
    if (img1.rows() != img2.rows() || img1.cols() != img2.cols()) {
        std::cerr << "Two images must have same size." << std::endl;
        return 0.0;
    }

    float mse = 0.0;
    int rows = img1.rows();
    int cols = img1.cols();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = img1(i, j) - img2(i, j);
            mse += diff * diff;
        }
    }
    mse /= (rows * cols);

    if (!getMSE) {
        float maxPixelValue = 255.0; // Assuming 8-bit images
        float psnr = 20.0 * log10(maxPixelValue) - 10.0 * log10(mse);
        return psnr;
    }
    else {
        return mse;
    }
}

float VQVEKLT::calculateCPSNR(const MatrixXf& img1_R, const MatrixXf& img1_G, const MatrixXf& img1_B, const MatrixXf& img2_R, const MatrixXf& img2_G, const MatrixXf& img2_B, bool getMSE)
{
    if (img1_R.rows() != img2_R.rows() || img1_R.cols() != img2_R.cols()) {
        std::cerr << "Two images must have same size." << std::endl;
        return 0.0;
    }

    float mse_r = calculatePSNR(img1_R, img2_R, true);
    float mse_g = calculatePSNR(img1_G, img2_G, true);
    float mse_b = calculatePSNR(img1_B, img2_B, true);
    float mse = (mse_r + mse_g + mse_b) / 3.0f;

    if (!getMSE) {
        float maxPixelValue = 255.0; // Assuming 8-bit images
        float psnr = 20.0 * log10(maxPixelValue) - 10.0 * log10(mse);
        return psnr;
    }
    else {
        return mse;
    }

    return 0.0f;
}

std::string VQVEKLT::extractFilename(const std::string& path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

std::string VQVEKLT::changeExtension(const std::string& filename, const std::string& newExtension)
{
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filename.substr(0, dotPos) + newExtension;
    }
    return filename + newExtension;
}

void VQVEKLT::afterAnalysis(const std::string& decoded_fname)
{
    if (floatImageGrayData == nullptr) {
        std::cerr << "Input gray scale image is null" << std::endl;
        return;
    }

    std::string fname = extractFilename(decoded_fname);
    std::string csv_path = analysisFolderName + "\\" + csvFileName;
    std::string size_str = std::to_string(width_bak) + "x" + std::to_string(height_bak);
    std::vector<std::string> row;
    float bpp = 0.0f;
    float psnr = 0.0f;
    if (numChannels == 1 || !COLOR_MODE) {
        MatrixXf input_img = Map<MatrixXf>(floatImageGrayData, width, height);
        loadInputImg(decodeFolderName + "\\" + changeExtension(decoded_fname, ".bmp"));
        convertToGrayScale();
        MatrixXf decoded_img = Map<MatrixXf>(floatImageGrayData, width, height);
        bpp = (fileSize * 8.0f) / decoded_img.size();
        psnr = calculatePSNR(input_img, decoded_img);

    }
    else {
        MatrixXf R_in, G_in, B_in;
        floatImageDataToRGB(R_in, G_in, B_in);
        MatrixXf R_de, G_de, B_de;
        loadInputImg(decodeFolderName + "\\" + changeExtension(decoded_fname, ".bmp"));
        floatImageDataToRGB(R_de, G_de, B_de);
        bpp = (fileSize * 8.0f) / (R_de.size() * 3.0f);
        psnr = calculateCPSNR(R_in, G_in, B_in, R_de, G_de, B_de);
    }

    row = {
        fname,
        size_str,
        std::to_string(!COLOR_MODE ? 1 : numChannels),
        std::to_string(_q),
        std::to_string(bpp), // bpp
        std::to_string(psnr), // psnr
        std::to_string(_eob_thresh_y),
        std::to_string(_eob_thresh_cbcr),
        std::to_string(veklt_rate(0)),
        std::to_string(veklt_rate(1)),
        std::to_string(veklt_rate(2)),
        std::to_string(dctN_rate(0)),
        std::to_string(dctN_rate(1)),
        std::to_string(dctN_rate(2)),
        std::to_string(dctN2_rate(0)),
        std::to_string(dctN2_rate(1)),
        std::to_string(dctN2_rate(2)),
        std::to_string(dctN4_rate(0)),
        std::to_string(dctN4_rate(1)),
        std::to_string(dctN4_rate(2)),
        std::to_string(encode_time),
        std::to_string(decode_time)
    };
    appendRowToCSV(csv_path, row, true);

    if (imageData != nullptr) {
        free(imageData);
        imageData = nullptr;
    }
    if (floatImageData != nullptr) {
        free(floatImageData);
        floatImageData = nullptr;
    }
    if (floatImageGrayData != nullptr) {
        free(floatImageGrayData);
        floatImageGrayData = nullptr;
    }
    if (floatImageCb != nullptr) {
        free(floatImageCb);
        floatImageCb = nullptr;
    }
    if (floatImageCr != nullptr) {
        free(floatImageCr);
        floatImageCr = nullptr;
    }
}

void VQVEKLT::genRandoms(int total, int num_to_pick, std::vector<int>& pickedNumbers)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, total);
    std::unordered_set<int> pickedSet;
    while (pickedSet.size() < num_to_pick) {
        int number = dis(gen);
        pickedSet.insert(number);
    }
    pickedNumbers = std::vector<int>(pickedSet.begin(), pickedSet.end());
}

void VQVEKLT::convertToGrayScale()
{
    floatImageGrayData = (float*)malloc(width * height * sizeof(float));

    bool err = false;
    if (floatImageGrayData != nullptr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (numChannels == 3) {
                    float r = floatImageData[idx * 3];
                    float g = floatImageData[idx * 3 + 1];
                    float b = floatImageData[idx * 3 + 2];
                    float grayValue = 0.2989f * r + 0.5870f * g + 0.1140f * b - 128.0f;
                    floatImageGrayData[idx] = grayValue;
                }
                else if (numChannels == 1) {
                    floatImageGrayData[idx] = floatImageData[idx] - 128.0f;
                }
                else {
                    err = true;
                    break;
                }
            }
        }
    }
    if (err) std::cerr << "Error: Unsupported numChannels " + numChannels << std::endl;
}

void VQVEKLT::convertToYCbCr()
{
    if (numChannels != 3) {
        std::cerr << "Error: Unsupported numChannels " + numChannels << std::endl;
    }

    floatImageGrayData = (float*)malloc(width * height * sizeof(float));
    width2 = width / 2;
    height2 = height / 2;
    int cbcrSize = width2 * height2;
    floatImageCb = (float*)malloc(cbcrSize * sizeof(float));
    floatImageCr = (float*)malloc(cbcrSize * sizeof(float));

    if (floatImageGrayData != nullptr && floatImageCb != nullptr && floatImageCr != nullptr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float r = floatImageData[idx * 3];
                float g = floatImageData[idx * 3 + 1];
                float b = floatImageData[idx * 3 + 2];
                floatImageGrayData[idx] = 0.2989f * r + 0.5870f * g + 0.1140f * b - 128.0f;
                if (x % 2 == 0 && y % 2 == 0) {
                    int idx2 = (y / 2) * width2 + (x / 2);
                    if (cbcrSize > idx2) {
                        floatImageCb[idx2] = -0.168736f * r - 0.331264f * g + 0.5f * b;
                        floatImageCr[idx2] = 0.5f * r - 0.418688f * g - 0.081312f * b;
                    }                    
                }
            }
        }
    }
}

void VQVEKLT::floatImageDataToRGB(MatrixXf& _R, MatrixXf& _G, MatrixXf& _B)
{
    if (numChannels != 3) {
        std::cerr << "Error: Unsupported numChannels " + numChannels << std::endl;
    }

    _R = MatrixXf(height_bak, width_bak);
    _G = MatrixXf(height_bak, width_bak);
    _B = MatrixXf(height_bak, width_bak);

    if (floatImageGrayData != nullptr && floatImageCb != nullptr && floatImageCr != nullptr) {
        for (int y = 0; y < height_bak; ++y) {
            for (int x = 0; x < width_bak; ++x) {
                int idx = y * width_bak + x;
                _R(y, x) = floatImageData[idx * 3];
                _G(y, x) = floatImageData[idx * 3 + 1];
                _B(y, x) = floatImageData[idx * 3 + 2];
            }
        }
    }
}

template <typename T>
T VQVEKLT::im2col(const T& image, int blockSize)
{
    int rows = image.rows();
    int cols = image.cols();
    int outputCols = cols * rows / (blockSize * blockSize);
    int outputRows = blockSize * blockSize;

    T result(outputRows, outputCols);
    int index = 0;
    for (int i = 0; i <= cols - blockSize; i += blockSize) {
        for (int j = 0; j <= rows - blockSize; j += blockSize) {
            for (int k = 0; k < blockSize; k++) {
                for (int l = 0; l < blockSize; l++) {
                    result(blockSize * k + l, index) = image(j + l, i + k);
                }
            }
            index += 1;
        }
    }

    return result;
}

template <typename T>
void VQVEKLT::col2im(T& cols, T& im, int blockWidth, int blockHeight)
{
    int blockRows = im.rows() / blockHeight;
    int cols_cols = cols.cols();
    for (int i = 0; i < cols_cols; i++) {
        int x = i % blockRows;
        int y = i / blockRows;
        im.block(blockHeight * x, blockWidth * y, blockHeight, blockWidth) 
            = Map<T>(cols.col(i).data(), blockWidth, blockHeight);
    }
}

void VQVEKLT::findEOB(const MatrixXf& aci, VectorXi& eobN, VectorXi& aco)
{
    int aci_cols = aci.cols();
    int aci_rows = aci.rows();
    eobN = VectorXi::Zero(aci_cols);
    aco = VectorXi::Zero(aci_rows * aci_cols);
    int lc = 0;
    int leng = 0;
    for (int colIndex = 0; colIndex < aci_cols; colIndex++) {
        int lastNonZeroIndex = -1;
        for (int rowIndex = 0; rowIndex < aci_rows; rowIndex++) {
            if (aci(rowIndex, colIndex) != 0) {
                lastNonZeroIndex = rowIndex;
            }
        }
        if (lastNonZeroIndex != -1) {
            eobN(colIndex) = lastNonZeroIndex;
            leng = lastNonZeroIndex + 1;
            int lb = lc;
            lc = lc + leng;
            aco.segment(lb, leng) = aci.col(colIndex).segment(0, leng).array().round().cast<int>();
        }
    }
    aco.conservativeResize(lc);
}

void VQVEKLT::onlyNonZeroElements(const VectorXi& seq, VectorXi& cdf, VectorXi& seq_ind)
{
    VectorXi nonZeroElements(seq.size());
    int count = 0;
    for (int i = 0; i < seq.size(); i++) {
        if (seq(i) != 0) {
            nonZeroElements(count++) = abs(seq(i));
        }
    }
    nonZeroElements.conservativeResize(count);
    int dum0, dum1;
    getCdfAndSeqIdx(nonZeroElements, cdf, seq_ind, dum0, dum1);
}

void VQVEKLT::getCdfAndSeqIdx(const VectorXi& seq, VectorXi& cdf, VectorXi& seq_ind, int& min, int& max)
{
    int mins = seq.minCoeff();
    int maxs = seq.maxCoeff();
    if (maxs - mins + 1 == 1) {
        cdf = VectorXi(1);
        cdf << seq.size();
        seq_ind = VectorXi::Zero(seq.size());
        min = mins;
        max = maxs;
        return;
    }

    std::map<int, int> histogram;
    for (int i = 0; i < seq.size(); i++) {
        if (seq(i) >= mins && seq(i) <= maxs) {
            histogram[seq(i)]++;
        }
    }

    cdf = VectorXi(maxs - mins + 1);
    int cumulativeCount = 0;
    for (int val = mins; val <= maxs; ++val) {
        if (histogram.find(val) != histogram.end()) {
            cumulativeCount += histogram[val];
        }
        cdf(val - mins) = cumulativeCount;
    }

    seq_ind = seq.array() - mins;
    min = mins;
    max = maxs;
}

void VQVEKLT::arithmeticEncode(const VectorXi& seq, std::vector<char>& code)
{
    VectorXi _cdf;
    VectorXi seq_ind;
    int mins, maxs;
    getCdfAndSeqIdx(seq, _cdf, seq_ind, mins, maxs);
    VectorXi cdf(_cdf.size() + 2);
    cdf(0) = mins; cdf(1) = maxs; cdf(2) = _cdf(0);
    int cdf_size = cdf.size();
    for (int i = 3; i < cdf_size; i++) {
        cdf(i) = _cdf(i - 2) - _cdf(i - 3);
    }
    VectorXi grBinArr;
    golombEncode(cdf, grBinArr);

    int maxBits = floor(log2(maxs + 1)) + 2;
    VectorXi seqBinArr = VectorXi::Ones(seq_ind.size() * maxBits);
    _arithmeticEncode(_cdf, seq_ind, seqBinArr);
        
    VectorXi binArr(grBinArr.size() + seqBinArr.size());
    binArr.head(grBinArr.size()) = grBinArr;
    binArr.tail(seqBinArr.size()) = seqBinArr;

    convertBinaryToChar(binArr, code);
}

int VQVEKLT::arithmeticDecode(const std::vector<char>& code, VectorXi& seq)
{
    VectorXi binaryArray;
    convertCharToBinary(code, binaryArray);

    VectorXi minMax(2);
    int next_index = golombDecode(binaryArray, minMax);
    int cdf_length = minMax(1) - minMax(0) + 1;
    VectorXi _cdf(cdf_length);

    VectorXi tmp = binaryArray.tail(binaryArray.size() - next_index);
    binaryArray = tmp;
    next_index = golombDecode(binaryArray, _cdf);
    int _cdf_size = _cdf.size();
    for (int i = 1; i < _cdf_size; i++) {
        _cdf(i) = _cdf(i - 1) + _cdf(i);
    }

    VectorXi cdf(_cdf.size() + 1);
    cdf.tail(_cdf.size()) = _cdf;

    tmp = binaryArray.tail(binaryArray.size() - next_index);
    binaryArray = tmp;
    VectorXi seq_ind;
    _arithmeticDecode(cdf, binaryArray, seq_ind);

    VectorXi sym = VectorXi::LinSpaced(cdf_length, minMax(0), minMax(1));
    seq = sym(seq_ind);

    return minMax(1);
}

void VQVEKLT::_arithmeticEncode(const VectorXi& cdf, const VectorXi& seq_ind, VectorXi& binaryArray)
{
    double mScale = 0;
    double mLow = 0;
    double mHigh = pow(2, 31) - 1;
    double g_Half = pow(2, 30);
    double g_FirstQuarter = pow(2, 29);
    double g_ThirdQuarter = pow(2, 29) * 3;
    int cl = -1;
    int end = cdf.size() - 1;

    int seq_ind_size = seq_ind.size();
    for (int i = 0; i < seq_ind_size; i++) {
        double mStep = floor((mHigh - mLow + 1) / cdf(end));
        mHigh = mLow + mStep * cdf(seq_ind(i)) - 1;
        if (seq_ind(i) != 0) {
            mLow = mLow + mStep * cdf(seq_ind(i) - 1);
        }
        while ((mHigh < g_Half) || (mLow >= g_Half)) {
            if (mHigh < g_Half) {
                cl++;
                binaryArray(cl) = 0;
                mLow = mLow * 2;
                mHigh = mHigh * 2 + 1;
                while (mScale > 0) {
                    cl++;
                    mScale--;
                }
            }
            else if (mLow >= g_Half) {
                cl++;
                mLow = 2 * (mLow - g_Half);
                mHigh = 2 * (mHigh - g_Half) + 1;
                while (mScale > 0) {
                    cl++;
                    binaryArray(cl) = 0;
                    mScale--;
                }
            }
        }
        while ((g_FirstQuarter <= mLow) && (mHigh < g_ThirdQuarter)) {
            mScale++;
            mLow = 2 * (mLow - g_FirstQuarter);
            mHigh = 2 * (mHigh - g_FirstQuarter) + 1;
        }
    }
    binaryArray.conservativeResize(cl + 2);
}

void VQVEKLT::_arithmeticDecode(const VectorXi& cdf, const VectorXi& binaryArray, VectorXi& seq_ind)
{
    double mLow = 0;
    double mHigh = pow(2, 31) - 1;
    double g_Half = pow(2, 30);
    double g_FirstQuarter = pow(2, 29);
    double g_ThirdQuarter = pow(2, 29) * 3;
    VectorXi bits(binaryArray.size() + 50); VectorXi zeros = VectorXi::Zero(50);
    bits << binaryArray, zeros;
    double mBuffer = 0.0;
    for (int i = 30; i >= 0; i--) {
        mBuffer += bits(30 - i) * pow(2, i);
    }
    int end = cdf.size() - 1;
    int code_index = 30; int ls = cdf(end);
    seq_ind = VectorXi::Zero(ls);
    for (int i = 0; i < ls; i++) {
        double mStep = floor((mHigh - mLow + 1) / cdf(end));
        double value = floor((mBuffer - mLow) / mStep);
        for (int j = 0; j < end; j++) {
            if (value >= cdf(j) && value < cdf(j + 1)) {
                seq_ind(i) = j;
                break;
            }
        }
        mHigh = mLow + mStep * cdf(seq_ind(i) + 1) - 1;
        if (seq_ind(i) != 0) {
            mLow = mLow + mStep * cdf(seq_ind(i));
        }
        while ((mHigh < g_Half) || (mLow >= g_Half)) {
            if (mHigh < g_Half) {
                mLow = 2 * mLow;
                mHigh = 2 * mHigh + 1;
                code_index++;
                if (code_index > bits.size() - 1) {
                    mBuffer = 2 * mBuffer;
                }
                else {
                    mBuffer = 2 * mBuffer + bits(code_index);
                }
            }
            else if (mLow >= g_Half) {
                mLow = 2 * (mLow - g_Half);
                mHigh = 2 * (mHigh - g_Half) + 1;
                code_index++;
                if (code_index > bits.size() - 1) {
                    mBuffer = 2 * (mBuffer - g_Half);
                }
                else {
                    mBuffer = 2 * (mBuffer - g_Half) + bits(code_index);
                }
            }
        }
        while ((g_FirstQuarter <= mLow) && (mHigh < g_ThirdQuarter)) {
            mLow = 2 * (mLow - g_FirstQuarter);
            mHigh = 2 * (mHigh - g_FirstQuarter) + 1;
            code_index++;
            if (code_index > bits.size() - 1) {
                mBuffer = 2 * (mBuffer - g_FirstQuarter);
            }
            else {
                mBuffer = 2 * (mBuffer - g_FirstQuarter) + bits(code_index);
            }
        }
    }
}

void VQVEKLT::indN2ari(const VectorXi& seq, int indN_size, std::vector<char>& code)
{
    if (seq.size() < 2) {
        code.resize(1);
        code[0] = 0;
        return;
    }
    VectorXi _cdf(1);
    int vekltCount = (seq.array() == 0).count();
    _cdf << vekltCount;
    VectorXi cdf(2);
    cdf << vekltCount, indN_size;
    VectorXi grBinArr;
    golombEncode(_cdf, grBinArr);

    int maxBits = 3;
    VectorXi seqBinArr = VectorXi::Ones(indN_size * maxBits);
    _arithmeticEncode(cdf, seq, seqBinArr);

    VectorXi binArr(grBinArr.size() + seqBinArr.size());
    binArr.head(grBinArr.size()) = grBinArr;
    binArr.tail(seqBinArr.size()) = seqBinArr;

    convertBinaryToChar(binArr, code);
}

void VQVEKLT::ari2indN(const std::vector<char>& code, int indN_size, VectorXi& seq)
{
    VectorXi binaryArray;
    convertCharToBinary(code, binaryArray);

    VectorXi _cdf(1);
    int next_index = golombDecode(binaryArray, _cdf);
    
    VectorXi tmp = binaryArray.tail(binaryArray.size() - next_index);
    binaryArray = tmp;
    VectorXi cdf(3);
    cdf << 0, _cdf(0), indN_size;
    _arithmeticDecode(cdf, binaryArray, seq);
}

void VQVEKLT::signAC2ari(const VectorXi& seq, int signAC_size, std::vector<char>& code)
{
    VectorXi cdf(3);
    int zeroCount = (seq.array() == 0).count();
    int oneCount = (seq.array() == 1).count();
    cdf << zeroCount, (zeroCount + oneCount), signAC_size;
    VectorXi grBinArr;
    golombEncode(cdf, grBinArr);

    int maxBits = 3;
    VectorXi seqBinArr = VectorXi::Ones(signAC_size * maxBits);
    _arithmeticEncode(cdf, seq, seqBinArr);

    VectorXi binArr(grBinArr.size() + seqBinArr.size());
    binArr.head(grBinArr.size()) = grBinArr;
    binArr.tail(seqBinArr.size()) = seqBinArr;

    convertBinaryToChar(binArr, code);
}

void VQVEKLT::ari2signAC(const std::vector<char>& code, VectorXi& seq)
{
    VectorXi binaryArray;
    convertCharToBinary(code, binaryArray);

    VectorXi _cdf(3);
    int next_index = golombDecode(binaryArray, _cdf);

    VectorXi tmp = binaryArray.tail(binaryArray.size() - next_index);
    binaryArray = tmp;
    VectorXi cdf(4);
    cdf << 0, _cdf(0), _cdf(1), _cdf(2);
    _arithmeticDecode(cdf, binaryArray, seq);
}

void VQVEKLT::golombEncode(const VectorXi& seq, VectorXi& binaryArray)
{
    int maxSymbol = seq.maxCoeff();
    int m = floor(log2(maxSymbol + 1));
    int symbolLength = 2 * m + 1;
    binaryArray = VectorXi(symbolLength * seq.size());

    int first = 0;
    int seq_size = seq.size();
    for (int i = 0; i < seq_size; ++i) {
        int n = floor(log2(seq(i) + 1));
        int leng = 2 * n + 1;
        VectorXi bits(leng); bits.setZero();
        bits(n) = 1;
        int d = seq(i) + 1 - pow(2, n);
        int e = floor(log2(d)) + 1;
        int max_n_e = std::max(n, e);
        VectorXi remain(max_n_e);
        int count = 0;
        for (int j = 1 - max_n_e; j <= 0; j++) {
            int val = floor(d * pow(2, j));
            remain(count++) = val % 2;
        }
        bits.segment(n + 1, max_n_e) = remain;
        binaryArray.segment(first, leng) = bits;
        first = first + leng;
    }
    binaryArray.conservativeResize(first);
}

int VQVEKLT::golombDecode(const VectorXi& binaryArray, VectorXi& seq)
{
    int j = 0;
    int seq_size = seq.size();
    for (int i = 0; i < seq_size; i++) {
        int symbol = 0;
        int length_M = 0;
        int x = 0;
        while (x < 1) {
            if (binaryArray(j) == 1) {
                if (length_M == 0) {
                    symbol = 0;
                    j++;
                    x = 1;
                }
                else {
                    VectorXi bits = binaryArray.segment(j + 1, length_M);
                    VectorXi powers(length_M);
                    for (int k = 0; k < length_M; k++) {
                        powers(k) = pow(2, length_M - 1 - k);
                    }
                    int info = bits.dot(powers);
                    symbol = pow(2, length_M) + info - 1;
                    j += length_M + 1;
                    length_M = 0;
                    x = 1;
                }
            }
            else { // binaryArray(j) == 0
                length_M++;
                j++;
            }
        }
        seq(i) = symbol;
    }
    return j;
}

void VQVEKLT::insertEOB(const MatrixXf& aci, VectorXi& aco)
{
    int lc = 0;
    int leng = 0;
    int aci_cols = aci.cols();
    for (int i = 0; i < aci_cols; i++) {
        VectorXi acii = aci.col(i).cast<int>();
        int lastNonZeroIndex = -1;
        for (int j = 0; j < acii.size(); j++) {
            if (acii(j) != 0) lastNonZeroIndex = j;
        }
        if (lastNonZeroIndex != -1) {
            leng = lastNonZeroIndex + 1;
            int lb = lc;
            aco.segment(lb, leng) = acii.segment(0, leng);
            lc = lc + leng;
            aco(lc) = tmpeobsym;
        }
        else {
            aco(lc) = tmpeobsym;
        }
        lc++;
    }
    aco.conservativeResize(lc);
}

void VQVEKLT::qtlevel3(VectorXi& indN, int widthN, int heightN, VectorXi& qtindN, VectorXi& qtindN2, VectorXi& qtindN4)
{
    MatrixXi _Inim = Map<MatrixXi>(indN.data(), heightN, widthN);
    int rem_height = _Inim.rows() % 4;
    int rem_width = _Inim.cols() % 4;
    int Inim_h = heightN - rem_height;
    int Inim_w = widthN - rem_width;
    MatrixXi Inim = _Inim.block(0, 0, Inim_h, Inim_w);
    MatrixXi Inim_left_down = _Inim.block(Inim_h, 0, rem_height, Inim_w);
    MatrixXi Inim_right_up = _Inim.block(0, Inim_w, Inim_h, rem_width);
    MatrixXi Inim_right_down = _Inim.block(Inim_h, Inim_w, rem_height, rem_width);

    MatrixXi Inimcol = im2col(Inim, 4);
    MatrixXi __Inimcolelse32 = MatrixXi::Zero(Inimcol.rows(), Inimcol.cols());
    VectorXi ind = Inimcol.colwise().sum();
    qtindN4 = VectorXi(ind.size());
    int c = 0;
    VectorXi v3 = VectorXi::Ones(Inimcol.rows()); v3 = v3 * 3;
    int ind_size = ind.size();
    for (int i = 0; i < ind_size; i++) {
        if (ind(i) == 16) {
            qtindN4(i) = 1;
            Inimcol.col(i) = v3;
        }
        else {
            qtindN4(i) = 0;
            __Inimcolelse32.col(c++) = Inimcol.col(i);
        }
    }
    MatrixXi _Inimcolelse32 = __Inimcolelse32.block(0, 0, __Inimcolelse32.rows(), c);
    VectorXi index(16);
    index << 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15;
    MatrixXi Inimcolelse32 = MatrixXi::Zero(_Inimcolelse32.rows(), _Inimcolelse32.cols());
    int index_size = index.size();
    for (int i = 0; i < index_size; i++) {
        Inimcolelse32.row(i) = _Inimcolelse32.row(index(i));
    }
    int Inimcolelse32_cols = Inimcolelse32.cols();
    MatrixXi Inimcol16(4, Inimcolelse32_cols * 4);
    for (int i = 0; i < Inimcolelse32_cols; i++) {
        Inimcol16.block(0,4*i,4,4) = Map<MatrixXi>(Inimcolelse32.col(i).data(), 4, 4);
    }
    ind = Inimcol16.colwise().sum();
    ind_size = ind.size();
    qtindN2 = VectorXi(ind_size);
    VectorXi v2 = VectorXi::Ones(Inimcol16.rows()); v2 = v2 * 2;
    for (int i = 0; i < ind_size; i++) {
        if (ind(i) == 4) {
            qtindN2(i) = 1;
            Inimcol16.col(i) = v2;
        }
        else {
            qtindN2(i) = 0;
        }
    }
    for (int i = 0; i < Inimcolelse32_cols; i++) {
        MatrixXi _tmp = Inimcol16.block(0, 4 * i, 4, 4);
        VectorXi tmp(16);
        for (int j = 0; j < index.size(); j++) {
            int idx = index(j);
            int x = idx % 4;
            int y = idx / 4;
            tmp(j) = _tmp(x, y);
        }
        Inimcolelse32.col(i) = tmp;
    }
    c = 0;
    int Inimcol_cols = Inimcol.cols();
    for (int i = 0; i < Inimcol_cols; i++) {
        if (qtindN4(i) == 0) {
            Inimcol.col(i) = Inimcolelse32.col(c++);
        }
    }

    MatrixXi _im = MatrixXi::Zero(Inim_h, Inim_w);
    col2im(Inimcol, _im, 4, 4);
    MatrixXi im = MatrixXi::Zero(heightN, widthN);
    im.block(0, 0, Inim_h, Inim_w) = _im;
    im.block(Inim_h, 0, rem_height, Inim_w) = Inim_left_down;
    im.block(0, Inim_w, Inim_h, rem_width) = Inim_right_up;
    im.block(Inim_h, Inim_w, rem_height, rem_width) = Inim_right_down;
    qtindN = Map<VectorXi>(im.data(), im.size());
}

void VQVEKLT::getSelected(const MatrixXf& input, const VectorXi& ind, const int val, MatrixXf& selected)
{
    selected = MatrixXf(input.rows(), (ind.array() == val).count());

    int c = 0;
    int size = ind.size();
    for (int i = 0; i < size; i++) {
        if (ind(i) == val) {
            selected.col(c++) = input.col(i);
        }
    }
}

void VQVEKLT::prepareCodingForVEKLT(const MatrixXf& y, VectorXi& out_Sign, VectorXi& out_AC)
{
    VectorXi Qac_VEKLT((y.rows() + 1) * y.cols()); Qac_VEKLT.setZero();
    insertEOB(y, Qac_VEKLT);
    VectorXi sign_VEKLT = Qac_VEKLT.array().sign();
    out_Sign = VectorXi(sign_VEKLT.size() - (Qac_VEKLT.array() == tmpeobsym).count());
    Qac_VEKLT = Qac_VEKLT.array().abs();
    int c = 0;
    int max = std::numeric_limits<int>::min();
    int Qac_VEKLT_size = Qac_VEKLT.size();
    for (int i = 0; i < Qac_VEKLT_size; i++) {
        if (Qac_VEKLT(i) != tmpeobsym) {
            out_Sign(c++) = sign_VEKLT(i) == -1 ? 2 : sign_VEKLT(i);
            if (max < Qac_VEKLT(i)) max = Qac_VEKLT(i);
        }
    }
    if (max == std::numeric_limits<int>::min()) {
        max = 1;
    }
    else {
        max = max + 1;
    }
    int veklt_eobsym = max;
    out_AC = VectorXi(Qac_VEKLT_size - (Qac_VEKLT.array() == 0).count());
    c = 0;
    for (int i = 0; i < Qac_VEKLT_size; i++) {
        if (Qac_VEKLT(i) != 0) {
            if (Qac_VEKLT(i) == tmpeobsym)
                out_AC(c++) = veklt_eobsym;
            else
                out_AC(c++) = Qac_VEKLT(i);
        }
    }
}

void VQVEKLT::prepareCodingForDCT(const MatrixXf& y, const int zSize, VectorXi& out_Sign, VectorXi& out_AC, VectorXi& out_DC)
{
    MatrixXf _Qac = y.block(1, 0, y.rows() - 1, y.cols());
    MatrixXf Qac = MatrixXf::Zero(_Qac.rows(), _Qac.cols());
    if (zSize == Z32_SIZE) {
        for (int i = 0; i < Z32_SIZE; i++) {
            Qac.row(Z32(i)) = _Qac.row(i);
        }
    }
    else if (zSize == Z16_SIZE) {
        for (int i = 0; i < Z16_SIZE; i++) {
            Qac.row(Z16(i)) = _Qac.row(i);
        }
    }
    else if (zSize == Z8_SIZE) {
        for (int i = 0; i < Z8_SIZE; i++) {
            Qac.row(Z8(i)) = _Qac.row(i);
        }
    }
    else {
        std::cerr << "Error: Unsupported size : " + zSize << std::endl;
    }

    VectorXi Qac_DCT = VectorXi::Zero((Qac.rows() + 1) * Qac.cols());
    int qac_eobsym = Qac.array().abs().maxCoeff() + 1;
    insertEOB(Qac, Qac_DCT);
    int qac_size = Qac_DCT.size();
    out_Sign = VectorXi::Zero(qac_size);
    int sign_c = 0;
    out_AC = VectorXi::Zero(qac_size);
    int abs_c = 0;
    for (int i = 0; i < qac_size; i++) {
        int val = Qac_DCT(i);
        if (val > 0) {
            if (val != tmpeobsym) {
                out_Sign(sign_c++) = 1;
                out_AC(abs_c++) = abs(val);
            }
            else {
                out_AC(abs_c++) = qac_eobsym;
            }
        }
        else if (val < 0) {
            out_Sign(sign_c++) = 2;
            out_AC(abs_c++) = abs(val);
        }
        else {
            out_Sign(sign_c++) = 0;
        }
    }
    out_Sign.conservativeResize(sign_c);
    out_AC.conservativeResize(abs_c);

    MatrixXf _Qdc = y.block(0, 0, 1, y.cols());
    int qdc_size = _Qdc.size();
    out_DC = VectorXi(qdc_size);
    for (int i = 0; i < qdc_size; i++) {
        out_DC(i) = round(_Qdc(i));
    }
}

void VQVEKLT::rind2code(const VectorXi& Rind, std::vector<char>& code)
{
    int lRind = Rind.size();
    VectorXi bits(12 * lRind);
    size_t bitsSize = bits.size();
    int c = 0;
    for (size_t i = 0; i < bitsSize; i += 12) {
        int binaryArray[12];
        int decimal = Rind(c++);
        for (int j = 11; j >= 0; --j) {
            binaryArray[j] = decimal & 1;
            decimal >>= 1;
        }
        bits.segment(i, 12) = Map<VectorXi>(binaryArray, 12);
    }
    convertBinaryToChar(bits, code);
}

void VQVEKLT::convertBinaryToChar(const VectorXi& binaryArray, std::vector<char>& charArray)
{
    size_t binarySize = binaryArray.size();
    size_t charArraySize = (binarySize + 7) / 8;

    charArray.resize(charArraySize);

    for (size_t i = 0; i < binarySize; i += 8) {
        char currentChar = 0;
        for (size_t j = 0; j < 8 && i + j < binarySize; ++j) {
            currentChar |= (binaryArray(i + j) == 1 ? 1 : 0) << (7 - j);
        }
        charArray[i / 8] = currentChar;
    }
}

void VQVEKLT::code2rind(const std::vector<char>& charArray, VectorXi& Rind)
{
    VectorXi binaryArray;
    convertCharToBinary(charArray, binaryArray);

    size_t binSize = binaryArray.size();
    size_t final_i = binSize % 12 == 0 ? binSize : (binSize - (binSize % 12));

    int c = 0;
    for (size_t i = 0; i < final_i; i += 12) {
        int decimal = 0;
        VectorXi bin = binaryArray.segment(i, 12);
        for (int j = 0; j < 12; ++j) {
            decimal = (decimal << 1) | bin(j);
        }
        Rind(c++) = decimal;
    }
}

void VQVEKLT::convertCharToBinary(const std::vector<char>& charArray, VectorXi& binaryArray)
{
    size_t binSize = charArray.size() * 8;
    binaryArray = VectorXi(binSize);

    size_t count = 0;
    for (char c : charArray) {
        for (int i = 7; i >= 0; --i) {
            bool bit = (c >> i) & 1;
            binaryArray(count++) = bit ? 1 : 0;
        }
    }
}

void VQVEKLT::decodeDCcoefficient(const char* buffer, const int min_qdc, int& start, int& end, int length, VectorXi& qdc)
{
    start = end;
    end = start + length;
    if (length != 0) {
        std::vector<char> some_code;
        some_code.assign(buffer + start, buffer + end);
        ari2dc(some_code, min_qdc, qdc);
    }
    else {
        qdc = VectorXi(1);
        qdc << min_qdc;
    }    
}

void VQVEKLT::decodeRindAndVQcoefficient(const char* buffer, int mode, int min_qvq, int& start, int& end, int rind_length, int qvq_length, VectorXi& Rind, VectorXi& qvq)
{
    start = end;
    end = start + rind_length;
    std::vector<char> rindCode;
    rindCode.assign(buffer + start, buffer + end);
    start = end;
    end = start + qvq_length;
    std::vector<char> qVqCode;
    qVqCode.assign(buffer + start, buffer + end);
    if (qVqCode.size() > 0) {
        ari2vq(qVqCode, min_qvq, qvq);
    }
    else {
        return;
    }
    if (rindCode.size() > 0) {
        if (mode == 0) {
            Rind = VectorXi(qvq.size());
            code2rind(rindCode, Rind);
        }
        else {
            arithmeticDecode(rindCode, Rind);
        }
    }
}

void VQVEKLT::decodeHeader(const char* buffer, int mode, int& start, int& end, VectorXi& header)
{
    if (mode == 0) {
        int header_length = buffer[0]; header_length += buffer[1];
        std::vector<char> header_code;
        start = 2;
        end = start + header_length;
        header_code.assign(buffer + start, buffer + end);
        VectorXi headerBinary;
        convertCharToBinary(header_code, headerBinary);

        header = VectorXi(26);
        golombDecode(headerBinary, header);
        numChannels = header(2);
    }
    else {
        start = end;
        int header_length = buffer[start]; header_length += buffer[start+1];
        std::vector<char> header_code;
        start = start + 2;
        end = start + header_length;
        header_code.assign(buffer + start, buffer + end);
        VectorXi headerBinary;
        convertCharToBinary(header_code, headerBinary);

        header = VectorXi(22);
        golombDecode(headerBinary, header);
    }
}

void VQVEKLT::decodeBinaryIndexImage(const char* buffer, int indN_size, int& start, int& end, int length, VectorXi& indN)
{
    start = end;
    end = start + length;
    std::vector<char> some_code;
    some_code.assign(buffer + start, buffer + end);
    if (some_code.size() > 0) {
        ari2indN(some_code, indN_size, indN);
    }
}

void VQVEKLT::decodeSignACcoefficient(const char* buffer, int& start, int& end, int length, VectorXi& signQac)
{
    start = end;
    end = start + length;
    std::vector<char> some_code;
    some_code.assign(buffer + start, buffer + end);
    if (some_code.size() > 0) {
        ari2signAC(some_code, signQac);
    }
}

int VQVEKLT::decodeAbsACcoefficient(const char* buffer, int& start, int& end, int length, VectorXi& absQac)
{
    start = end;
    end = start + length;
    std::vector<char> some_code;
    some_code.assign(buffer + start, buffer + end);
    int max = 0;
    if (some_code.size() > 0) {
        max = arithmeticDecode(some_code, absQac); // eobsym
    }
    return max;
}

void VQVEKLT::colMreshape(const VectorXi& abs_seq, const VectorXi& sign_seq, int eobsym, MatrixXf& colMat)
{
    int abs_seq_idx = 0;
    int sign_seq_idx = 0;
    std::vector<int> absBlock;
    int colMat_cols = colMat.cols();
    for (int i = 0; i < colMat_cols; i++) {
        absBlock.clear();
        while (abs_seq(abs_seq_idx++) != eobsym) {
            absBlock.push_back(abs_seq(abs_seq_idx-1));
        }
        int non0Num = absBlock.size();
        if (non0Num < 1) continue;
        int j = 0;
        int k = 0;
        while (k < non0Num) {
            if (sign_seq(sign_seq_idx) == 0) {
                sign_seq_idx++;
                j++;
            }
            else {
                int sign = sign_seq(sign_seq_idx++) == 2 ? -1 : 1;
                colMat(j++, i) = sign * absBlock[k++];
            }
        }
    }
}

void VQVEKLT::dc2ari(const VectorXi& qdc, std::vector<char>& code, int& min)
{
    VectorXi diff(qdc.size()); diff(0) = qdc(0);
    int diff_size = diff.size();
    for (int i = 1; i < diff_size; i++) {
        diff(i) = qdc(i) - qdc(i - 1);
    }
    min = diff.minCoeff();
    diff = diff.array() - min;
    arithmeticEncode(diff, code);
}

void VQVEKLT::vq2ari(const VectorXi& qvq, std::vector<char>& code, int& min)
{
    VectorXi qvq2;
    min = qvq.minCoeff();
    qvq2 = qvq.array() - min;
    arithmeticEncode(qvq2, code);
}

int VQVEKLT::getMinQdcFlag(int min, bool exist)
{
    if (!exist) {
        return 2;
    }
    else {
        if (min < 0) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

void VQVEKLT::matrixCal_By_ARow(const MatrixXf& A, const MatrixXf& B, MatrixXf& C)
{
    int howMany = A.rows();
    int rows_per_thread = howMany / num_threads;
    int rows_remain = howMany % num_threads;
    std::vector<std::thread> threads;
    int start_row = 0;
    int end_row = -1;

    MatrixCal_By_ARow thread_obj(A, B);

    int Arows = A.rows();
    int Bcols = B.cols();
    C = MatrixXf(Arows, Bcols);
    std::vector<MatrixXf> block_vec(num_threads);
    for (int i = 0; i < num_threads; i++) {
        start_row = end_row + 1;
        end_row = start_row + rows_per_thread - 1;
        MatrixXf block;
        block_vec[i] = block;
        threads.push_back(std::thread(thread_obj, start_row, rows_per_thread, &block_vec[i]));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    start_row = 0;
    for (int i = 0; i < num_threads; i++) {
        C.block(start_row, 0, rows_per_thread, Bcols) = block_vec[i];
        start_row += rows_per_thread;
    }
    int Acols = A.cols();
    if (rows_remain > 0) {
        C.block(start_row, 0, rows_remain, Bcols) = A.block(start_row, 0, rows_remain, Acols) * B;
    }
}

void VQVEKLT::matrixCal_By_BCol(const MatrixXf& A, const MatrixXf& B, MatrixXf& C)
{
    int howMany = B.cols();
    int cols_per_thread = howMany / num_threads;
    int cols_remain = howMany % num_threads;
    std::vector<std::thread> threads;
    int start_col = 0;
    int end_col = -1;

    MatrixCal_By_BCol thread_obj(A, B);

    int Arows = A.rows();
    int Bcols = B.cols();
    C = MatrixXf(Arows, Bcols);
    std::vector<MatrixXf> block_vec(num_threads);
    for (int i = 0; i < num_threads; i++) {
        start_col = end_col + 1;
        end_col = start_col + cols_per_thread - 1;
        MatrixXf block;
        block_vec[i] = block;
        threads.push_back(std::thread(thread_obj, start_col, cols_per_thread, &block_vec[i]));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    int Brows = B.rows(); // = Acols;
    start_col = 0;
    for (int i = 0; i < num_threads; i++) {
        C.block(0, start_col, Arows, cols_per_thread) = block_vec[i];
        start_col += cols_per_thread;
    }
    if (cols_remain > 0) {
        C.block(0, start_col, Arows, cols_remain) = A * B.block(0, start_col, Brows, cols_remain);
    }
}

void VQVEKLT::ari2dc(const std::vector<char>& code, int min, VectorXi& qdc)
{
    VectorXi diff;
    arithmeticDecode(code, diff);
    diff = diff.array() + min;
    int diff_size = diff.size();
    qdc = VectorXi(diff_size); qdc(0) = diff(0);
    for (int i = 1; i < diff_size; i++) {
        qdc(i) = qdc(i - 1) + diff(i);
    }
}

void VQVEKLT::ari2vq(const std::vector<char>& code, int min, VectorXi& qvq)
{
    VectorXi qvq2;
    arithmeticDecode(code, qvq2);
    qvq = qvq2.array() + min;
}

void VQVEKLT::appendRowToCSV(const std::string& filename, const std::vector<std::string>& row, bool append)
{
    std::ofstream file;
    if (append) {
        file = std::ofstream(filename, std::ios::app); // Open file in append mode
    }
    else {
        file = std::ofstream(filename, std::ios::out);
    }
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < row.size(); ++i) {
        file << row[i];
        if (i != row.size() - 1) {
            file << ",";
        }
    }
    file << std::endl;

    file.close();
}

std::string VQVEKLT::getFileNameWithExtension(const std::string& filename)
{
    std::istringstream iss(filename);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(iss, token, '.')) {
        tokens.push_back(token);
    }

    if (tokens.size() > 1) {
        return tokens[0] + ".vjp";
    }
    else {
        return filename;
    }
}

void VQVEKLT::saveDecodedImage(const MatrixXf& Rim, int mode, int padN_h, int padN_w, int N)
{
    std::string bmpFname = changeExtension(inputVjpName, ".bmp");
    bool paddedOrNot = false;
    MatrixXf RimFinal;
    if (padN_h % N != 0 || padN_w % N != 0) {
        paddedOrNot = true;
        height -= (padN_h % N);
        width -= (padN_w % N);
        RimFinal = Rim.block(0, 0, height, width);
    }
    if (numChannels == 1 || !COLOR_MODE) {
        height_bak = height;
        width_bak = width;
        saveY2BMP(paddedOrNot ? RimFinal : Rim, bmpFname);
    }
    else {
        if (mode == 0) {
            Y_bak = paddedOrNot ? RimFinal : Rim;
            height_bak = height;
            width_bak = width;
        }
        else if (mode == 1) {
            Cb_bak = paddedOrNot ? RimFinal : Rim;
        }
        else if (mode == 2) {
            MatrixXf Cr;
            if (paddedOrNot)
                scaleUpMatrix(RimFinal, Cr, width_bak, height_bak);
            else 
                scaleUpMatrix(Rim, Cr, width_bak, height_bak);
            MatrixXf Cb;
            scaleUpMatrix(Cb_bak, Cb, width_bak, height_bak);
            MatrixXf R, G, B;
            YCbCr2RGB(Y_bak, Cb, Cr, R, G, B);
            saveRGB2BMP(R, G, B, bmpFname);
        }
        else {
            std::cerr << "saveDecodedImage : invalid mode" << std::endl;
        }
    }
}

void VQVEKLT::enCode(const std::string& fname, float q, int eob_thresh_y, int eob_thresh_cbcr)
{   
    _eob_thresh_y = eob_thresh_y;
    _eob_thresh_cbcr = eob_thresh_cbcr;

    loadInputImg(fname);
#if ELAPSED_TIME_TOTAL
    auto start = std::chrono::high_resolution_clock::now();
#endif
    if (numChannels == 1 || !COLOR_MODE) {
        convertToGrayScale();
        enCodeYCbCr(q, 0);
    }
    else {
        convertToYCbCr();
        enCodeYCbCr(q, 0);
        enCodeYCbCr(q, 1);
        enCodeYCbCr(q, 2);
    }
#if ELAPSED_TIME_TOTAL
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    encode_time = duration.count();
#endif
}

void VQVEKLT::enCodeYCbCr(float q, int mode)
{
    int N;
    MatrixXf inputMat, EV, DCTVN, DCTVN2, DCTVN4;
    if (mode == 0) {
        N = 8;
        inputMat = Map<MatrixXf>(floatImageGrayData, width, height);
        EV = EVmatrixes;
        DCTVN = DCTV8;
        DCTVN2 = DCTV16;
        DCTVN4 = DCTV32;
    }
    else if (mode == 1 || mode == 2) {
        N = 4;
        width = width2;
        height = height2;
        if (mode == 1) { // Cb
            inputMat = Map<MatrixXf>(floatImageCb, width, height);
            EV = EV_Cb;
        }
        else { // Cr
            inputMat = Map<MatrixXf>(floatImageCr, width, height);
            EV = EV_Cr;
        }
        DCTVN = DCTV4;
        DCTVN2 = DCTV8;
        DCTVN4 = DCTV16;
    }
    else {
        std::cerr << "enCodeYCbCr : invalid mode" << std::endl;
        return;
    }
    float Qv = 16 * q;
    float Qd = Qv;
    _q = q;

    int heightN = height / N;
    int widthN = width / N;
    int padN_h = N - height % N;
    int padN_w = N - width % N;
    MatrixXf x = inputMat.transpose();

    if (padN_h != N || padN_w != N) {
        padN_h = padN_h % N;
        padN_w = padN_w % N;
        MatrixXf new_x = MatrixXf::Zero(height + padN_h, width + padN_w);
        new_x.block(0, 0, height, width) = x;
        RowVectorXf pad_left_down = x.row(height - 1);
        MatrixXf left_down = pad_left_down.replicate(padN_h, 1);
        VectorXf pad_right_up = x.col(width - 1);
        MatrixXf right_up = pad_right_up.replicate(1, padN_w);
        MatrixXf right_down = MatrixXf::Ones(padN_h, padN_w) * x(height - 1, width - 1);
        new_x.block(height, 0, padN_h, width) = left_down;
        new_x.block(0, width, height, padN_w) = right_up;
        new_x.block(height, width, padN_h, padN_w) = right_down;
        x = new_x;
        height += padN_h;
        width += padN_w;
        heightN = height / N;
        widthN = width / N;
    }
    MatrixXf xN = im2col(x, N);
    MatrixXf y = xN;
    MatrixXf DCTVNy = ((DCTVN * y).array() / Qd).round();
    MatrixXf Qdc = DCTVNy.block(0, 0, 1, DCTVNy.cols());
    MatrixXf _Qac = DCTVNy.block(1, 0, DCTVNy.rows() - 1, DCTVNy.cols());
    MatrixXf Qac = MatrixXf::Zero(_Qac.rows(), _Qac.cols());
    if (mode == 0) {
        for (int i = 0; i < Z8_SIZE; i++) {
            Qac.row(Z8(i)) = _Qac.row(i);
        }
    }
    else {
        for (int i = 0; i < Z4_SIZE; i++) {
            Qac.row(Z4(i)) = _Qac.row(i);
        }
    }    
    int QacCols = Qac.cols();
    VectorXi eob_DCTN = VectorXi::Zero(QacCols);
    VectorXi Qac_DCT = VectorXi::Zero(Qac.rows() * QacCols);
    findEOB(Qac, eob_DCTN, Qac_DCT);
    VectorXi indN = VectorXi::Ones(eob_DCTN.size());
    if (mode == 0) {
        indN = (eob_DCTN.array() > _eob_thresh_y).select(0, indN);
    }
    else {
        indN = (eob_DCTN.array() > _eob_thresh_cbcr).select(0, indN);
    }
    int total_vqveklt = indN.size() - indN.count();
    veklt_rate(mode) = total_vqveklt * 1.0f / indN.size();

    int leng_col = N * N;
    MatrixXf vqveklt_y(leng_col, total_vqveklt);
    MatrixXf vqveklt_Qdc(leng_col, total_vqveklt);
    MatrixXf vqveklt_xN(leng_col, total_vqveklt);
    int vqveklt_c = 0;
    int ycols = y.cols();
    for (int i = 0; i < ycols; i++) {
        if (indN(i) == 0) {
            vqveklt_y.col(vqveklt_c) = y.col(i);
            MatrixXf tmp_Qdc = MatrixXf::Ones(leng_col, 1); tmp_Qdc *= Qdc(i);
            vqveklt_Qdc.col(vqveklt_c) = tmp_Qdc;
            vqveklt_xN.col(vqveklt_c) = xN.col(i);
            vqveklt_c += 1;
        }
    }
    vqveklt_y = vqveklt_y - (1.0 / N) * vqveklt_Qdc * Qd;
    MatrixXf norm_vqveklt_y = vqveklt_y.array().square().colwise().sum().sqrt();
    MatrixXf repmat_norm_vqveklt_y(leng_col, norm_vqveklt_y.size());
    for (int i = 0; i < leng_col; i++) {
        repmat_norm_vqveklt_y.row(i) = norm_vqveklt_y;
    }
    MatrixXf feature = vqveklt_y.array() / repmat_norm_vqveklt_y.array();
    MatrixXf RR;
    if (mode == 0) {
        RR = R;
    }
    else {
        RR = R4;
    }
    MatrixXf Rt = RR.transpose();
    MatrixXf RtFea;
    if (feature.cols() > 1000) {
        if (mode == 0) {
            if (Rt.rows() < feature.cols()) {
                matrixCal_By_BCol(Rt, feature, RtFea);
            }
            else {
                matrixCal_By_ARow(Rt, feature, RtFea);
            }
        }
        else {
            matrixCal_By_BCol(Rt, feature, RtFea);
        }
    }
    else {
        RtFea = Rt * feature;
    }
    VectorXi Rind(RtFea.cols());
    for (int i = 0; i < total_vqveklt; i++) {
        RtFea.col(i).maxCoeff(&Rind[i]);
    }
    int Rind_size = Rind.size();
    VectorXf vq(Rind_size);
    for (int i = 0; i < Rind_size; i++) {
        vq(i) = (vqveklt_xN.col(i).array() * RR.col(Rind(i)).array()).sum();
        vqveklt_y.col(i) = vqveklt_y.col(i) - vq(i) * RR.col(Rind(i));
    }
    VectorXi Qvq = (vq.array() / Qv).round().cast<int>();
    MatrixXf QVQVEKLTy(leng_col - 2, total_vqveklt);
    int EVrows = N * N;
    int EVcols = N * N - 2;
    for (int i = 0; i < total_vqveklt; i++) {
        MatrixXf EVt = EV.block(0, Rind(i) * EVcols, EVrows, EVcols).transpose();
        QVQVEKLTy.col(i) = (EVt * vqveklt_y.col(i) / Qv).array().round();
    }
    VectorXi out_sign_VEKLT;
    VectorXi out_Qac_VEKLT;
    if (total_vqveklt > 0) {
        prepareCodingForVEKLT(QVQVEKLTy, out_sign_VEKLT, out_Qac_VEKLT);
    }
    VectorXi qtindN;
    VectorXi qtindN2;
    VectorXi qtindN4;
    qtlevel3(indN, widthN, heightN, qtindN, qtindN2, qtindN4);
    int N4 = N * 4;
    int remN4_h = x.rows() % N4;
    int remN4_w = x.cols() % N4;
    MatrixXf xx = x.block(0, 0, height - remN4_h, width - remN4_w);

    // DCTN4
    MatrixXf xN4 = im2col(xx, N4);
    MatrixXf xN4sel;
    getSelected(xN4, qtindN4, 1, xN4sel);
    bool encode_N4 = false;
    VectorXi out_sign_DCTN4;
    VectorXi out_Qdc_DCTN4;
    VectorXi out_Qac_DCTN4;
    dctN4_rate(mode) = 0.0f;
    if (xN4sel.cols() > 0) {
        dctN4_rate(mode) = xN4sel.cols() * 16.0f / indN.size();
        encode_N4 = true;
        MatrixXf QDCTVN4y;
        if (xN4sel.cols() > 1000) {
            if (DCTVN4.rows() > xN4sel.cols()) {
                matrixCal_By_ARow(DCTVN4, xN4sel, QDCTVN4y);
            }
            else {
                matrixCal_By_BCol(DCTVN4, xN4sel, QDCTVN4y);
            }
            QDCTVN4y = (QDCTVN4y.array() / Qd).round();
        }
        else {
            QDCTVN4y = ((DCTVN4 * xN4sel).array() / Qd).round();
        }
        if (mode == 0) {
            prepareCodingForDCT(QDCTVN4y, Z32_SIZE, out_sign_DCTN4, out_Qac_DCTN4, out_Qdc_DCTN4);
        }
        else {
            prepareCodingForDCT(QDCTVN4y, Z16_SIZE, out_sign_DCTN4, out_Qac_DCTN4, out_Qdc_DCTN4);
        }
    }

    // DCTN2
    MatrixXf xN4selelse;
    getSelected(xN4, qtindN4, 0, xN4selelse);
    MatrixXf xN2 = MatrixXf::Zero(4 * N * N, xN4selelse.cols() * 4);
    int N2 = N * 2;
    int xN4selelseCols = xN4selelse.cols();
    for (int i = 0; i < xN4selelseCols; i++) {
        MatrixXf tmp = Map<MatrixXf>(xN4selelse.col(i).data(), N4, N4);
        xN2.block(0, 4 * i, xN2.rows(), 4) = im2col(tmp, N2);
    }
    MatrixXf xN2sel;
    getSelected(xN2, qtindN2, 1, xN2sel);
    bool encode_N2 = false;
    VectorXi out_sign_DCTN2;
    VectorXi out_Qdc_DCTN2;
    VectorXi out_Qac_DCTN2;
    dctN2_rate(mode) = 0.0f;
    if (xN2sel.cols() > 0) {
        dctN2_rate(mode) = xN2sel.cols() * 4.0f / indN.size();
        encode_N2 = true;
        MatrixXf QDCTVN2y = ((DCTVN2 * xN2sel).array() / Qd).round();
        if (mode == 0) {
            prepareCodingForDCT(QDCTVN2y, Z16_SIZE, out_sign_DCTN2, out_Qac_DCTN2, out_Qdc_DCTN2);
        }
        else {
            prepareCodingForDCT(QDCTVN2y, Z8_SIZE, out_sign_DCTN2, out_Qac_DCTN2, out_Qdc_DCTN2);
        }
    }
    // DCTN
    MatrixXf QacN;
    getSelected(Qac, qtindN, 1, QacN);
    int sign_c = 0;
    VectorXi out_sign_DCTN;
    int abs_c = 0;
    VectorXi out_Qac_DCTN;
    dctN_rate(mode) = 0.0f;
    if (QacN.cols() > 0) {
        dctN_rate(mode) = QacN.cols() * 1.0f / indN.size();
        VectorXi Qac_DCTN = VectorXi::Zero(QacN.rows() * (QacN.cols() + 1));
        int qac_eobsym = QacN.array().abs().maxCoeff() + 1;
        insertEOB(QacN, Qac_DCTN);
        int qac_size = Qac_DCTN.size();
        out_sign_DCTN = VectorXi::Zero(qac_size);
        out_Qac_DCTN = VectorXi::Zero(qac_size);
        for (int i = 0; i < qac_size; i++) {
            int val = Qac_DCTN(i);
            if (val > 0) {
                if (val != tmpeobsym) {
                    out_sign_DCTN(sign_c++) = 1;
                    out_Qac_DCTN(abs_c++) = abs(val);
                }
                else {
                    out_Qac_DCTN(abs_c++) = qac_eobsym;
                }
            }
            else if (val < 0) {
                out_sign_DCTN(sign_c++) = 2;
                out_Qac_DCTN(abs_c++) = abs(val);
            }
            else {
                out_sign_DCTN(sign_c++) = 0;
            }
        }
    }
    out_sign_DCTN.conservativeResize(sign_c);
    out_Qac_DCTN.conservativeResize(abs_c);
    VectorXi QdcSmallelse((qtindN.array() == 0 || qtindN.array() == 1).count());
    int c = 0;
    int qtindN_size = qtindN.size();
    for (int i = 0; i < qtindN_size; i++) {
        if (qtindN(i) == 0 || qtindN(i) == 1) {
            QdcSmallelse(c++) = Qdc(i);
        }
    }
    double out_q = q * 1000;
    std::vector<char> out_Rind_code;
    if (Rind.size() > 0) {
        if (mode == 0) {
            rind2code(Rind, out_Rind_code);
        }
        else {
            arithmeticEncode(Rind, out_Rind_code);
        }
    }
    std::vector<char> out_vq_code;
    int min_Qvq = 0;
    if (Qvq.size() > 0) {
        vq2ari(Qvq, out_vq_code, min_Qvq);
    }
    std::vector<char> out_indN_code;
    int indN_size = indN.size();
    if (total_vqveklt != indN_size && total_vqveklt != 0) {
        indN2ari(indN, indN_size, out_indN_code);
    }
    std::vector<char> out_QdcN_code; int min_QdcN = 0;
    if (QdcSmallelse.size() > 0) {
        dc2ari(QdcSmallelse, out_QdcN_code, min_QdcN);
    }
    std::vector<char> out_QdcN2_code;
    int min_QdcN2 = 0;
    if (encode_N2) {
        if (out_Qdc_DCTN2.size() > 1) {
            dc2ari(out_Qdc_DCTN2, out_QdcN2_code, min_QdcN2);
        }
        else {
            min_QdcN2 = out_Qdc_DCTN2(0);
        }
    }
    std::vector<char> out_QdcN4_code;
    int min_QdcN4 = 0;
    if (encode_N4) {
        if (out_Qdc_DCTN4.size() > 1) {
            dc2ari(out_Qdc_DCTN4, out_QdcN4_code, min_QdcN4);
        }
        else {
            min_QdcN4 = out_Qdc_DCTN4(0);
        }
    }
    std::vector<char> out_sign_QacN_code;
    int out_sign_DCTN_size = out_sign_DCTN.size();
    if (out_sign_DCTN_size > 0) {
        signAC2ari(out_sign_DCTN, out_sign_DCTN_size, out_sign_QacN_code);
    }
    std::vector<char> out_sign_QacN2_code;
    int out_sign_DCTN2_size = out_sign_DCTN2.size();
    if (encode_N2 && out_sign_DCTN2_size > 0) {
        signAC2ari(out_sign_DCTN2, out_sign_DCTN2_size, out_sign_QacN2_code);
    }
    std::vector<char> out_sign_QacN4_code;
    int out_sign_DCTN4_size = out_sign_DCTN4.size();
    if (encode_N4 && out_sign_DCTN4_size > 0) {
        signAC2ari(out_sign_DCTN4, out_sign_DCTN4_size, out_sign_QacN4_code);
    }
    std::vector<char> out_sign_veklt_code;
    int out_sign_VEKLT_size = out_sign_VEKLT.size();
    if (out_sign_VEKLT_size > 0) {
        signAC2ari(out_sign_VEKLT, out_sign_VEKLT_size, out_sign_veklt_code);
    }
    std::vector<char> out_abs_QacN_code;
    if (out_sign_DCTN.size() > 0) {
        arithmeticEncode(out_Qac_DCTN, out_abs_QacN_code);
    }
    std::vector<char> out_abs_QacN2_code;
    if (encode_N2 && out_sign_DCTN2.size() > 0) {
        arithmeticEncode(out_Qac_DCTN2, out_abs_QacN2_code);
    }
    std::vector<char> out_abs_QacN4_code;
    if (encode_N4 && out_sign_DCTN4.size() > 0) {
        arithmeticEncode(out_Qac_DCTN4, out_abs_QacN4_code);
    }
    std::vector<char> out_abs_veklt_code;
    if (out_sign_VEKLT.size() > 0) {
        arithmeticEncode(out_Qac_VEKLT, out_abs_veklt_code);
    }

    VectorXi header;
    if (mode == 0) {
        header = VectorXi(26);
        header(0) = height - (padN_h % N);
        header(1) = width - (padN_w % N);
        header(2) = numChannels;
        header(3) = out_q;
        header(4) = min_Qvq < 0 ? 1 : 0; header(5) = abs(min_Qvq);
        header(6) = min_QdcN < 0 ? 1 : 0; header(7) = abs(min_QdcN);
        header(8) = getMinQdcFlag(min_QdcN2, encode_N2); header(9) = abs(min_QdcN2);
        header(10) = getMinQdcFlag(min_QdcN4, encode_N4); header(11) = abs(min_QdcN4);
        header(12) = out_QdcN_code.size(); header(13) = out_QdcN2_code.size();
        header(14) = out_QdcN4_code.size();
        header(15) = out_Rind_code.size(); header(16) = out_vq_code.size();
        header(17) = out_indN_code.size();
        header(18) = out_sign_QacN_code.size(); header(19) = out_sign_QacN2_code.size();
        header(20) = out_sign_QacN4_code.size(); header(21) = out_sign_veklt_code.size();
        header(22) = out_abs_QacN_code.size(); header(23) = out_abs_QacN2_code.size();
        header(24) = out_abs_QacN4_code.size(); header(25) = out_abs_veklt_code.size();
    }
    else {
        header = VectorXi(22);
        header(0) = min_Qvq < 0 ? 1 : 0; header(1) = abs(min_Qvq);
        header(2) = min_QdcN < 0 ? 1 : 0; header(3) = abs(min_QdcN);
        header(4) = getMinQdcFlag(min_QdcN2, encode_N2); header(5) = abs(min_QdcN2);
        header(6) = getMinQdcFlag(min_QdcN4, encode_N4); header(7) = abs(min_QdcN4);
        header(8) = out_QdcN_code.size(); header(9) = out_QdcN2_code.size();
        header(10) = out_QdcN4_code.size();
        header(11) = out_Rind_code.size(); header(12) = out_vq_code.size();
        header(13) = out_indN_code.size();
        header(14) = out_sign_QacN_code.size(); header(15) = out_sign_QacN2_code.size();
        header(16) = out_sign_QacN4_code.size(); header(17) = out_sign_veklt_code.size();
        header(18) = out_abs_QacN_code.size(); header(19) = out_abs_QacN2_code.size();
        header(20) = out_abs_QacN4_code.size(); header(21) = out_abs_veklt_code.size();
    }

    VectorXi headerBinary;
    golombEncode(header, headerBinary);

    std::vector<char> header_code;
    convertBinaryToChar(headerBinary, header_code);

    std::vector<char> header_length_code(2);
    int header_length = header_code.size();
    std::bitset<16> header_length_bin = header_length;
    std::bitset<8> binary_first(header_length_bin.to_string().substr(0, 8));
    header_length_code[0] = static_cast<char>(binary_first.to_ulong());
    std::bitset<8> binary_second(header_length_bin.to_string().substr(8, 8));
    header_length_code[1] = static_cast<char>(binary_second.to_ulong());

    std::string out_fname = getFileNameWithExtension(inputImgName);
    std::string savePath = encodeFolderName + "\\" + out_fname;
    std::ofstream output_file;
    int modeDiff = 0;
    if (mode == 0) {
        output_file = std::ofstream(savePath, std::ios::binary | std::ios::trunc);
    }
    else {
        output_file = std::ofstream(savePath, std::ios::binary | std::ios::app);
        modeDiff = 4;
    }
    if (output_file.is_open()) {
        output_file.write(header_length_code.data(), 2);
        output_file.write(header_code.data(), header_length);

        output_file.write(out_QdcN_code.data(), header(12 - modeDiff));
        output_file.write(out_QdcN2_code.data(), header(13 - modeDiff));
        output_file.write(out_QdcN4_code.data(), header(14 - modeDiff));

        output_file.write(out_Rind_code.data(), header(15 - modeDiff));
        output_file.write(out_vq_code.data(), header(16 - modeDiff));
        output_file.write(out_indN_code.data(), header(17 - modeDiff));

        output_file.write(out_sign_QacN_code.data(), header(18 - modeDiff));
        output_file.write(out_sign_QacN2_code.data(), header(19 - modeDiff));
        output_file.write(out_sign_QacN4_code.data(), header(20 - modeDiff));
        output_file.write(out_sign_veklt_code.data(), header(21 - modeDiff));

        output_file.write(out_abs_QacN_code.data(), header(22 - modeDiff));
        output_file.write(out_abs_QacN2_code.data(), header(23 - modeDiff));
        output_file.write(out_abs_QacN4_code.data(), header(24 - modeDiff));
        output_file.write(out_abs_veklt_code.data(), header(25 - modeDiff));

        output_file.close();
    }
    else {
        std::cerr << "Error opening file for writing." << std::endl;
    }
}

void VQVEKLT::deCode(const std::string& fname)
{
#if ELAPSED_TIME_TOTAL
    auto start_clock = std::chrono::high_resolution_clock::now();
#endif
    inputVjpName = fname;
    std::string inputPath = "encoded\\" + fname;
    std::ifstream inputFile(inputPath.c_str(), std::ios::binary);
    if (!inputFile) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return;
    }
    inputFile.seekg(0, std::ios::end);
    fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    char* buffer = new char[fileSize];
    inputFile.read(buffer, fileSize);
    inputFile.close();

    int start = 0; int end = 0;
    VectorXi header;
    decodeHeader(buffer, 0, start, end, header);
    if (numChannels == 1 || !COLOR_MODE) {
        deCodeYCbCr(buffer, header, 0, start, end);
    }
    else {
        deCodeYCbCr(buffer, header, 0, start, end);
        
        decodeHeader(buffer, 1, start, end, header);
        deCodeYCbCr(buffer, header, 1, start, end);
        
        decodeHeader(buffer, 2, start, end, header);
        deCodeYCbCr(buffer, header, 2, start, end);
    }
#if ELAPSED_TIME_TOTAL
    auto end_clock = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - start_clock);
    decode_time = duration.count();
#endif
}

void VQVEKLT::deCodeYCbCr(const char* buffer, const VectorXi& header, int mode, int& start, int& end)
{
    int N;
    MatrixXf _EV, DCTVN, DCTVN2, DCTVN4, RR;
    int modeDiff = 0;
    if (mode == 0) {
        N = 8;
        _EV = EVmatrixes;
        DCTVN = DCTV8;
        DCTVN2 = DCTV16;
        DCTVN4 = DCTV32;
        RR = R;

        height = header(0);
        width = header(1);
        _q = header(3) / 1000.0;
    }
    else if (mode == 1 || mode == 2) {
        N = 4;
        if (mode == 1) { // Cb
            width2 = width / 2;
            height2 = height / 2;
            width = width2;
            height = height2;
            _EV = EV_Cb;
        }
        else { // Cr
            _EV = EV_Cr;
        }
        DCTVN = DCTV4;
        DCTVN2 = DCTV8;
        DCTVN4 = DCTV16;
        RR = R4;
        modeDiff = 4;
    }
    else {
        std::cerr << "deCodeYCbCr : invalid mode" << std::endl;
        return;
    }
    int padN_h = N - height % N;
    int padN_w = N - width % N;
    height += (padN_h % N);
    width += (padN_w % N);
    int block_num = (height / N) * (width / N);
    float Qv = 16 * _q;
    float Qd = Qv;

    int min_Qvq = (header(4 - modeDiff) == 1 ? -1 : 1) * header(5 - modeDiff);
    int min_QdcN = (header(6 - modeDiff) == 1 ? -1 : 1) * header(7 - modeDiff);
    int min_QdcN2_flag = header(8 - modeDiff);
    int min_QdcN2 = header(9 - modeDiff);
    if (min_QdcN2_flag == 1) {
        min_QdcN2 *= -1;
    }
    int min_QdcN4_flag = header(10 - modeDiff);
    int min_QdcN4 = header(11 - modeDiff);
    if (min_QdcN4_flag == 1) {
        min_QdcN4 *= -1;
    }

    VectorXi qDcSmallelse;
    decodeDCcoefficient(buffer, min_QdcN, start, end, header(12 - modeDiff), qDcSmallelse);
    VectorXi qDcN2;
    if (min_QdcN2_flag != 2) {
        decodeDCcoefficient(buffer, min_QdcN2, start, end, header(13 - modeDiff), qDcN2);
    }
    VectorXi qDcN4;
    if (min_QdcN4_flag != 2) {
        decodeDCcoefficient(buffer, min_QdcN4, start, end, header(14 - modeDiff), qDcN4);
    }
    VectorXi Rind, qVq;
    decodeRindAndVQcoefficient(buffer, mode, min_Qvq, start, end, header(15 - modeDiff), header(16 - modeDiff), Rind, qVq);
    VectorXi indN;
    if (Rind.size() == 0) {
        indN = VectorXi::Ones(block_num);
    }
    else if (Rind.size() == block_num) {
        indN = VectorXi::Zero(block_num);
    }
    else {
        decodeBinaryIndexImage(buffer, block_num, start, end, header(17 - modeDiff), indN);
    }
    VectorXi signQacN;
    decodeSignACcoefficient(buffer, start, end, header(18 - modeDiff), signQacN);
    VectorXi signQacN2;
    if (min_QdcN2_flag != 2) {
        decodeSignACcoefficient(buffer, start, end, header(19 - modeDiff), signQacN2);
    }
    VectorXi signQacN4;
    if (min_QdcN4_flag != 2) {
        decodeSignACcoefficient(buffer, start, end, header(20 - modeDiff), signQacN4);
    }
    VectorXi signVeklt;
    decodeSignACcoefficient(buffer, start, end, header(21 - modeDiff), signVeklt);
    VectorXi absQacN;
    int eobN = decodeAbsACcoefficient(buffer, start, end, header(22 - modeDiff), absQacN);
    VectorXi absQacN2; int eobN2 = std::numeric_limits<int>::min();
    if (min_QdcN2_flag != 2) {
        eobN2 = decodeAbsACcoefficient(buffer, start, end, header(23 - modeDiff), absQacN2);
    }
    VectorXi absQacN4; int eobN4 = std::numeric_limits<int>::min();
    if (min_QdcN4_flag != 2) {
        eobN4 = decodeAbsACcoefficient(buffer, start, end, header(24 - modeDiff), absQacN4);
    }
    VectorXi absVeklt;
    int eobVeklt = decodeAbsACcoefficient(buffer, start, end, header(25 - modeDiff), absVeklt);
    VectorXi qtindN;
    VectorXi qtindN2;
    VectorXi qtindN4;
    widthN = width / N;
    heightN = height / N;
    qtlevel3(indN, widthN, heightN, qtindN, qtindN2, qtindN4);
    int count_qDcN = (qtindN.array() == 1).count();
    int count_qDcVeklt = (qtindN.array() == 0).count();
    int c = 0;
    VectorXi indQdcSmallelse(count_qDcN + count_qDcVeklt);
    int qtindN_size = qtindN.size();
    for (int i = 0; i < qtindN_size; i++) {
        if (qtindN(i) == 0 || qtindN(i) == 1) {
            indQdcSmallelse(c++) = qtindN(i);
        }
    }
    VectorXi qDcN(count_qDcN); int c1 = 0;
    VectorXi qDcVeklt(count_qDcVeklt); int c2 = 0;
    int indQdcSmallelse_size = indQdcSmallelse.size();
    for (int i = 0; i < indQdcSmallelse_size; i++) {
        if (indQdcSmallelse(i) == 1) {
            qDcN(c1++) = qDcSmallelse(i);
        }
        else {
            qDcVeklt(c2++) = qDcSmallelse(i);
        }
    }
    MatrixXf _rQacN = MatrixXf::Zero(N * N - 1, count_qDcN);
    MatrixXf rQacN;
    int signQacN_size = signQacN.size();
    if (signQacN_size > 0) {
        colMreshape(absQacN, signQacN, eobN, _rQacN);
        rQacN = MatrixXf(_rQacN.rows(), _rQacN.cols());
        if (mode == 0) {
            for (int i = 0; i < Z8_SIZE; i++) {
                rQacN.row(i) = _rQacN.row(Z8(i));
            }
        }
        else {
            for (int i = 0; i < Z4_SIZE; i++) {
                rQacN.row(i) = _rQacN.row(Z4(i));
            }
        }
    }
    else {
        rQacN = _rQacN;
    }

    MatrixXf rQacN2;
    if (min_QdcN2_flag != 2) {
        int count_qDcN2 = (qtindN2.array() == 1).count();
        MatrixXf _rQacN2 = MatrixXf::Zero(4 * N * N - 1, count_qDcN2);
        if (signQacN2.size() > 0) {
            colMreshape(absQacN2, signQacN2, eobN2, _rQacN2);
            rQacN2 = MatrixXf(_rQacN2.rows(), _rQacN2.cols());
            if (mode == 0) {
                for (int i = 0; i < Z16_SIZE; i++) {
                    rQacN2.row(i) = _rQacN2.row(Z16(i));
                }
            }
            else {
                for (int i = 0; i < Z8_SIZE; i++) {
                    rQacN2.row(i) = _rQacN2.row(Z8(i));
                }
            }
        }
        else {
            rQacN2 = _rQacN2;
        }
    }
    MatrixXf rQacN4; int count_qDcN4 = 0;
    if (min_QdcN4_flag != 2) {
        count_qDcN4 = (qtindN4.array() == 1).count();
        MatrixXf _rQacN4 = MatrixXf::Zero(16 * N * N - 1, count_qDcN4);
        if (signQacN4.size() > 0) {
            colMreshape(absQacN4, signQacN4, eobN4, _rQacN4);
            rQacN4 = MatrixXf(_rQacN4.rows(), _rQacN4.cols());
            if (mode == 0) {
                for (int i = 0; i < Z32_SIZE; i++) {
                    rQacN4.row(i) = _rQacN4.row(Z32(i));
                }
            }
            else {
                for (int i = 0; i < Z16_SIZE; i++) {
                    rQacN4.row(i) = _rQacN4.row(Z16(i));
                }
            }
        }
        else {
            rQacN4 = _rQacN4;
        }
    }
    MatrixXf rVeklt = MatrixXf::Zero(N * N - 2, count_qDcVeklt);
    if (signVeklt.size() > 0) {
        colMreshape(absVeklt, signVeklt, eobVeklt, rVeklt);
    }
    MatrixXf RQN;
    if (count_qDcN > 0) {
        MatrixXf rQN(N * N, count_qDcN);
        rQN << qDcN.transpose().cast<float>(), rQacN;
        RQN = DCTVN.transpose() * rQN * Qd; RQN = RQN.array().round();
    }
    MatrixXf RQN2;
    if (min_QdcN2_flag != 2) {
        MatrixXf rQN2(rQacN2.rows() + 1, rQacN2.cols());
        rQN2 << qDcN2.transpose().cast<float>(), rQacN2;
        RQN2 = DCTVN2.transpose() * rQN2 * Qd; RQN2 = RQN2.array().round();
    }
    MatrixXf RQN4;
    if (min_QdcN4_flag != 2) {
        MatrixXf rQN4(rQacN4.rows() + 1, rQacN4.cols());
        rQN4 << qDcN4.transpose().cast<float>(), rQacN4;
        RQN4 = DCTVN4.transpose() * rQN4 * Qd; RQN4 = RQN4.array().round();
    }
    MatrixXf RVQVEKLT;
    int EVrows = N * N;
    int EVcols = N * N - 2;
    if (count_qDcVeklt > 0) {
        RVQVEKLT = MatrixXf::Zero(N * N, count_qDcVeklt);
        int qVq_size = qVq.size();
        for (int i = 0; i < qVq_size; i++) {
            MatrixXf EV = _EV.block(0, Rind(i) * EVcols, EVrows, EVcols);
            RVQVEKLT.col(i) = (Qv * EV * rVeklt.col(i)) + (Qv * qVq(i) * RR.col(Rind(i))) + (Qd * (1.0 / N) * qDcVeklt(i) * MatrixXf::Ones(N * N, 1));
        }
        RVQVEKLT = RVQVEKLT.array().round();
    }
    c1 = 0; c2 = 0;
    MatrixXf Ry = MatrixXf::Zero(N * N, width * height / (N * N));
    int Ry_cols = Ry.cols();
    for (int i = 0; i < Ry_cols; i++) {
        if (qtindN(i) == 1) {
            Ry.col(i) = RQN.col(c1++);
        }
        else if (qtindN(i) == 0) {
            Ry.col(i) = RVQVEKLT.col(c2++);
        }
    }
    MatrixXf _Rim = MatrixXf::Zero(height, width);
    col2im(Ry, _Rim, N, N);
    if (min_QdcN2_flag == 2 && min_QdcN4_flag == 2) {
        saveDecodedImage(_Rim, mode, padN_h, padN_w, N);
        if (mode == 0 && (numChannels == 1 || !COLOR_MODE)) {
            delete[] buffer;
        }
        else {
            if (mode == 2)
                delete[] buffer;
        }
        return;
    }

    int N4 = N * 4;
    int rem_height = _Rim.rows() % N4;
    int rem_width = _Rim.cols() % N4;
    int _Rim_Blank_h = height - rem_height;
    int _Rim_Blank_w = width - rem_width;
    MatrixXf _Rim_Blank = _Rim.block(0, 0, _Rim_Blank_h, _Rim_Blank_w);

    MatrixXf xN4; MatrixXf xN4selelse;
    if (min_QdcN4_flag != 2) {
        xN4 = im2col(_Rim_Blank, N4);
        if (min_QdcN2_flag != 2) {
            xN4selelse = MatrixXf::Zero(xN4.rows(), qtindN4.size() - count_qDcN4);
            c1 = 0; c2 = 0;
            int qtindN4_size = qtindN4.size();
            for (int i = 0; i < qtindN4_size; i++) {
                if (qtindN4(i) == 1) {
                    xN4.col(i) = RQN4.col(c1++);
                }
                else {
                    xN4selelse.col(c2++) = xN4.col(i);
                }
            }
        }
        else {
            c1 = 0;
            int qtindN4_size = qtindN4.size();
            for (int i = 0; i < qtindN4_size; i++) {
                if (qtindN4(i) == 1) {
                    xN4.col(i) = RQN4.col(c1++);
                }
            }
        }
    }
    else {
        xN4selelse = im2col(_Rim_Blank, N * 4);
    }

    if (min_QdcN2_flag != 2) {
        int xN2rows = 4 * N * N;
        MatrixXf xN2 = MatrixXf::Zero(xN2rows, xN4selelse.cols() * 4);
        int xN4selelse_cols = xN4selelse.cols();
        for (int i = 0; i < xN4selelse_cols; i++) {
            MatrixXf tmp = Map<MatrixXf>(xN4selelse.col(i).data(), 4 * N, 4 * N);
            xN2.block(0, 4 * i, xN2rows, 4) = im2col(tmp, 2 * N);
        }
        c = 0;
        int qtindN2_size = qtindN2.size();
        for (int i = 0; i < qtindN2_size; i++) {
            if (qtindN2(i) == 1) {
                xN2.col(i) = RQN2.col(c++);
            }
        }
        for (int i = 0; i < xN4selelse_cols; i++) {
            MatrixXf block = xN2.block(0, 4 * i, xN2rows, 4);
            MatrixXf _tmp = MatrixXf::Zero(4 * N, 4 * N);
            col2im(block, _tmp, 2 * N, 2 * N);
            VectorXf tmp = Map<VectorXf>(_tmp.data(), 16 * N * N);
            xN4selelse.col(i) = tmp;
        }
        if (min_QdcN4_flag != 2) {
            c = 0;
            int qtindN4_size = qtindN4.size();
            for (int i = 0; i < qtindN4_size; i++) {
                if (qtindN4(i) == 0) {
                    xN4.col(i) = xN4selelse.col(c++);
                }
            }
        }
        else {
            xN4 = xN4selelse;
        }
    }

    MatrixXf Rim_Filled = MatrixXf::Zero(_Rim_Blank_h, _Rim_Blank_w);
    col2im(xN4, Rim_Filled, 4 * N, 4 * N);
    MatrixXf Rim = MatrixXf::Zero(height, width);
    Rim.block(0, 0, _Rim_Blank_h, _Rim_Blank_w) = Rim_Filled;
    Rim.block(_Rim_Blank_h, 0, rem_height, _Rim_Blank_w) = _Rim.block(_Rim_Blank_h, 0, rem_height, _Rim_Blank_w);
    Rim.block(0, _Rim_Blank_w, _Rim_Blank_h, rem_width) = _Rim.block(0, _Rim_Blank_w, _Rim_Blank_h, rem_width);
    Rim.block(_Rim_Blank_h, _Rim_Blank_w, rem_height, rem_width) = _Rim.block(_Rim_Blank_h, _Rim_Blank_w, rem_height, rem_width);
    saveDecodedImage(Rim, mode, padN_h, padN_w, N);
    if (mode == 0 && (numChannels == 1 || !COLOR_MODE)) {
        delete[] buffer;
    }
    else {
        if (mode == 2)
            delete[] buffer;
    }
}
