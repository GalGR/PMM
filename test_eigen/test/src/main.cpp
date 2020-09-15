#include "data_shared_path.h"

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define STRINGIZE(X) #X

template <typename Scalar>
struct PMMGeodesicsData {
    size_t rows;
    size_t cols;
    template <decltype(Eigen::RowMajor) Major>
    struct Vertices {
        Eigen::Matrix<Scalar, -1, -1, Major> X;
        Eigen::Matrix<Scalar, -1, -1, Major> Y;
        Eigen::Matrix<Scalar, -1, -1, Major> Z;

        void resize(size_t rows, size_t cols) {
            X.resize(rows, cols);
            Y.resize(rows, cols);
            Z.resize(rows, cols);
        }

        friend std::ostream &operator << (std::ostream &out, const Vertices<Major> &o) {
            return out << STRINGIZE(PMMGeodesicsData<Scalar>::Vertices<Major>::X) ":" << std::endl << o.X << std::endl << STRINGIZE(PMMGeodesicsData<Scalar>::Vertices<Major>::Y) ":" << std::endl << o.Y << std::endl << STRINGIZE(PMMGeodesicsData<Scalar>::Vertices<Major>::Z) ":" << std::endl << o.Z << std::endl;
        }
    };
    Vertices<Eigen::RowMajor> V_row;
    Vertices<Eigen::ColMajor> V_col;

    friend std::ostream &operator << (std::ostream &out, const PMMGeodesicsData<Scalar> &o) {
        return out << STRINGIZE(PMMGeodesicsData<Scalar>::V_row) ":" << std::endl << o.V_row << std::endl << STRINGIZE(PMMGeodesicsData<Scalar>::V_col) ":" << std::endl << o.V_col << std::endl;
    }
};
PMMGeodesicsData<double> data;

template <typename T> std::string type_name();

int main(int argc, char *argv[]) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> row_major;
    row_major << 1, 2, 3, 4;
    std::cout << STRINGIZE(row_major) ":" << std::endl;
    std::cout << row_major << std::endl;
    std::cout << STRINGIZE(row_major.data()) ":" << std::endl;
    std::cout << row_major.data() << std::endl;
    Eigen::Matrix<double, 2, 2, Eigen::ColMajor> col_major;
    std::cout << STRINGIZE(col_major) " = " STRINGIZE(row_major) << std::endl;
    col_major = row_major;
    std::cout << STRINGIZE(col_major) ":" << std::endl;
    std::cout << col_major << std::endl;
    std::cout << STRINGIZE(col_major.data()) ":" << std::endl;
    std::cout << col_major.data() << std::endl;

    // Change (2,2) in row_major from 4 to 9
    Eigen::Map<Eigen::Array<double, 4, 1> > row_major_arr(row_major.data());
    std::cout << STRINGIZE(row_major_arr) ":" << std::endl;
    std::cout << row_major_arr << std::endl;
    row_major_arr(3) = 9;
    std::cout << STRINGIZE(row_major_arr) " = " STRINGIZE(9) << std::endl;
    std::cout << STRINGIZE(row_major_arr) ":" << std::endl;
    std::cout << row_major_arr << std::endl;
    std::cout << STRINGIZE(row_major) ":" << std::endl;
    std::cout << row_major << std::endl;
    Eigen::Array4d new_row_major_arr = Eigen::Map<Eigen::Array<double, 4, 1> >(row_major.data());
    std::cout << STRINGIZE(new_row_major_arr) ":" << std::endl;
    std::cout << new_row_major_arr << std::endl;
    new_row_major_arr(0) = 20;
    std::cout << STRINGIZE(new_row_major_arr) " = " STRINGIZE(20) << std::endl;
    std::cout << STRINGIZE(new_row_major_arr) ":" << std::endl;
    std::cout << new_row_major_arr << std::endl;
    std::cout << STRINGIZE(row_major) << std::endl;
    std::cout << row_major << std::endl;
    row_major.setConstant(1.0 / 0.0);
    std::cout << STRINGIZE(row_major.setConstant(1.0 / 0.0)) << std::endl;
    std::cout << STRINGIZE(row_major) ":" << std::endl;
    std::cout << row_major << std::endl;

    std::cout << data << std::endl;

    std::cout << std::endl << std::endl;

    Eigen::RowVector2d row_vec;
    row_vec << 1, 2;
    std::cout << STRINGIZE(row_vec) ":" << std::endl;
    std::cout << row_vec << std::endl;
    Eigen::Vector2d col_vec;
    col_vec << 2, 1;
    std::cout << STRINGIZE(col_vec) ":" << std::endl;
    std::cout << col_vec << std::endl;
    double res = row_vec * col_vec;
    std::cout << STRINGIZE(auto) " " STRINGIZE(res) " = " STRINGIZE(row_vec) " * " STRINGIZE(col_vec);
    std::cout << STRINGIZE(res) ":" << std::endl;
    std::cout << res << std::endl;
    // std::cout << STRINGIZE(res) " type:" << std::endl;
    // std::cout << type_name<decltype(res)>() << std::endl;
    // double d_res = res;

    std::cout << std::endl << std::endl;

    Eigen::Array4d arr;
    arr << 1, 2, 3, 4;
    std::cout << STRINGIZE(arr) ":" << std::endl;
    std::cout << arr << std::endl;
    auto cmp = arr > 2.0;
    std::cout << STRINGIZE(auto) " " STRINGIZE(cmp) " = " STRINGIZE(arr) " > " STRINGIZE(2.0) << std::endl;
    std::cout << STRINGIZE(cmp) ":" << std::endl;
    std::cout << cmp.all() << std::endl;

    // Test boost::program_options
    // ---------------------------

    std::string model_name;
    size_t img_size;
    int harmonic_const;
    size_t N_iters;
    std::vector<size_t> start_source;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("model,m", po::value<std::string>(&model_name), "model")
        ("length,l", po::value<size_t>(&img_size), "geometric image length")
        ("harmonic,H", po::value<int>(&harmonic_const)->default_value(1), "harmonic parameterization constant")
        ("iterations,i", po::value<size_t>(&N_iters)->default_value(1), "number of PMM iterations")
        ("source,s", po::value<std::vector<size_t> >(&start_source)->multitoken(), "start source vertices")
    ;
    po::positional_options_description p_opt;
    // p_opt.add("model", 1);
    // p_opt.add("source", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p_opt).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    if (vm.count("model,m") && vm.count("length,l")) {
        std::cerr << "'model' and 'length' are required arguments" << std::endl;
        std::cout << desc << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << STRINGIZE(model_name) ":\t\t" << model_name << std::endl;
    std::cout << STRINGIZE(img_size) ":\t\t" << img_size << std::endl;
    std::cout << STRINGIZE(harmonic_const) ":\t\t" << harmonic_const << std::endl;
    std::cout << STRINGIZE(N_iters) ":\t\t" << N_iters << std::endl;
    std::cout << STRINGIZE(start_source) ":\t\t";
    for (const auto &i : start_source) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;

    return 0;
}