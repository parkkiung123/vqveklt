#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Windows.h"
#include "VQVEKLT.h"

int main(int argc, char* argv[]) {
    VQVEKLT v;
    std::string dirPath = "images";
    v.listImagePath(dirPath);
    double length = v.imagePath.size();
    double percent = 0;
    std::vector<float> q_vec = { 0.5, 0.75, 1, 1.5, 2, 2.5, 3 };
    for (int i = 0; i < length; i++) {
        percent = ((i + 1) / length) * 100.0;
        std::string inputPath = v.imagePath[i];
        std::string fname = v.extractFilename(inputPath);
        for (int j = 0; j < q_vec.size(); j++) {
            v.enCode(inputPath, q_vec[j], 10, 10);
            std::string encoded_fname = v.changeExtension(fname, ".vjp");
            v.deCode(encoded_fname);
            v.afterAnalysis(fname);
        }
        std::cout << (i+1) << ":" << fname << ":" << percent << "%" << std::endl;
    }
    Beep(1500, 5000);
    return 0;
}