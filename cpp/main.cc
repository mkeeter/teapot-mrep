#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define MALLOC_CHECKS

// Based on https://github.com/stulp/tutorials/blob/master/test.md
#ifdef MALLOC_CHECKS
    #define EIGEN_RUNTIME_NO_MALLOC
    #define ENTERING_REAL_TIME_CRITICAL_CODE() Eigen::internal::set_is_malloc_allowed(false)
    #define EXITING_REAL_TIME_CRITICAL_CODE() Eigen::internal::set_is_malloc_allowed(true)
#else
    #define ENTERING_REAL_TIME_CRITICAL_CODE()
    #define EXITING_REAL_TIME_CRITICAL_CODE()
#endif

#include <Eigen/Dense>

using namespace Eigen;

using MatrixXv = Matrix<Vector3d, Dynamic, Dynamic>;

static const size_t MAX_N_CHOOSE_K = 10;
static double FACTORIAL[MAX_N_CHOOSE_K] = {0.0};
static double N_CHOOSE_K[MAX_N_CHOOSE_K][MAX_N_CHOOSE_K] = {0.0};

// Builds the `N_CHOOSE_K` table for fast lookup
void init() {
    FACTORIAL[0] = 1.0;
    for (size_t n=1; n < MAX_N_CHOOSE_K; ++n) {
        FACTORIAL[n] = FACTORIAL[n - 1] * n;
    }
    for (size_t n=0; n < MAX_N_CHOOSE_K; ++n) {
        for (size_t k=0; k < MAX_N_CHOOSE_K; ++k) {
            N_CHOOSE_K[n][k] = FACTORIAL[n] /
                               double(FACTORIAL[k] * FACTORIAL[n - k]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

double bernstein(size_t i, size_t degree, double u) {
    return N_CHOOSE_K[degree][i] * pow(u, i) * pow(1 - u, degree - i);
}

double bernstein_derive(size_t i, size_t degree, double u) {
    const double a = (i != 0)
        ? i * pow(u, i - 1) * pow(1 - u, degree - i)
        : 0.0;
    const double b = (i != degree)
        ? -pow(u, i) * (degree - i) * pow(1 - u, degree - i - 1)
        : 0.0;
    return N_CHOOSE_K[degree][i] * (a + b);
}

////////////////////////////////////////////////////////////////////////////////

std::vector<MatrixXv> parse_bpt(std::stringstream& buffer) {
    std::string line;
    getline(buffer, line);
    int count = std::stoi(line);

    // How does anyone write a safe parser in this language?!
    std::vector<MatrixXv> patches;
    for (int i=0; i < count; ++i) {
        getline(buffer, line);

        size_t pos;
        const size_t n = std::stoi(line, &pos);
        const size_t m = std::stoi(line.substr(pos + 1));

        MatrixXv mat(n + 1, m + 1);
        for (int j=0; j < n + 1; ++j) {
            for (int k=0; k < m + 1; ++k) {
                getline(buffer, line);

                const double x = std::stod(line, &pos);
                line = line.substr(pos + 1);
                const double y = std::stod(line, &pos);
                line = line.substr(pos + 1);
                const double z = std::stod(line, &pos);

                const Vector3d v(x, y, z);
                mat(j, k) = v;
            }
        }
        patches.push_back(mat);
    }
    return patches;
}

////////////////////////////////////////////////////////////////////////////////

MatrixXd S_v(const MatrixXv& b, const Vector2i& v) {
    const size_t degree1 = b.rows() - 1;
    const size_t degree2 = b.cols() - 1;

    const size_t stride = (v[0] + 1) * (v[1] + 1);
    MatrixXd out = MatrixXd::Zero(
        (degree1 + v[0] + 1) * (degree2 + v[1] + 1),
        4 * stride);
    // This isn't the most efficient way to build the matrix, but it's a
    // one-for-one copy of the known-good Numpy code.
    for (int axis=0; axis < 4; ++axis) {
        for (size_t k=0; k < v[0] + 1; ++k) {
            const auto v_k = N_CHOOSE_K[v[0]][k];
            for (size_t l=0; l < v[1] + 1; ++l) {
                const auto v_l = N_CHOOSE_K[v[1]][l];
                for (size_t i=0; i < degree1 + 1; ++i) {
                    for (size_t j=0; j < degree2 + 1; ++j) {
                        // B_{i+k} * B_{j+l}
                        const auto row = (j + l) + (i + k) * (v[1] + degree2 + 1);
                        const double c = axis
                            ? b(i, j)[axis - 1]
                            : 1.0;
                        const auto col = l + k * (v[1] + 1) + axis * stride;
                        out(row, col) +=
                            v_k * v_l * N_CHOOSE_K[degree1][i] * N_CHOOSE_K[degree2][j]
                            / (N_CHOOSE_K[v[0] + degree1][i + k] * N_CHOOSE_K[v[1] + degree2][j + l])
                            * c;
                    }
                }
            }
        }
    }
    return out;
}

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    Scratch(size_t rows, size_t cols)
        : stride(std::max(cols, rows) + 1)
    {
        for (size_t r=0; r < stride; ++r) {
            for (size_t c=0; c < stride; ++c) {
                svdsU.push_back(JacobiSVD<MatrixXd>(r, c,  ComputeFullU));
                svdsV.push_back(JacobiSVD<MatrixXd>(r, c,  ComputeFullV));
                mats.push_back(MatrixXd::Zero(r, c));
                tmps.push_back(MatrixXd::Zero(r, c));
            }
            eigs.push_back(GeneralizedEigenSolver<MatrixXd>(r));
            svdsUV.push_back(JacobiSVD<MatrixXd>(r, 1, ComputeThinU | ComputeThinV));
        }
    }

    size_t index(size_t r, size_t c) const {
        return c + r * stride;
    }

    MatrixXd& mat(size_t r, size_t c) {
        return mats[index(r, c)];
    }

    MatrixXd& tmp(size_t r, size_t c) {
        return tmps[index(r, c)];
    }

    MatrixXd& transpose(const MatrixXd& in) {
        MatrixXd& target = mat(in.cols(), in.rows());
        assert(&in != &target);
        target.noalias() = in.transpose();
        return target;
    }

    template<typename M, typename N>
    MatrixXd& matmul(const M& a, const N& b) {
        MatrixXd& target = tmps[index(a.rows(), b.cols())];
        assert((void*)&target != (void*)&a);
        assert((void*)&target != (void*)&b);
        target.noalias() = a * b;
        MatrixXd& out = mat(a.rows(), b.cols());
        out.noalias() = target;
        return out;
    }

    template<typename M>
    MatrixXd& rightCols(const M& a, size_t c) {
        auto& target = mat(a.rows(), c);
        assert((void*)&target != (void*)&a);

        target.noalias() = a.rightCols(c);
        return target;
    }

    template<typename M>
    JacobiSVD<MatrixXd>& svdU(const M& a) {
        MatrixXd& target = tmps[index(a.rows(), a.cols())];
        target.noalias() = a;
        auto& s = svdsU[target.cols() + stride * target.rows()];
        s.compute(target, ComputeFullU);
        return s;
    }

    template<typename M>
    JacobiSVD<MatrixXd>& svdV(const M& a) {
        MatrixXd& target = tmps[index(a.rows(), a.cols())];
        target.noalias() = a;
        auto& s = svdsV[target.cols() + stride * target.rows()];
        s.compute(target, ComputeFullV);
        return s;
    }

    double solve(const MatrixXd& A, const MatrixXd& B) {
        assert(A.cols() == 1);
        assert(B.cols() == 1);
        assert(A.rows() == B.rows());
        auto& s = svdsUV[A.rows()];
        s.compute(A, ComputeThinU | ComputeThinV);
        auto sol = s.solve(B);
        return sol(0,0);
    }

    const size_t stride;
    std::vector<JacobiSVD<MatrixXd>> svdsU;
    std::vector<JacobiSVD<MatrixXd>> svdsV;
    std::vector<JacobiSVD<MatrixXd>> svdsUV;

    // Somewhat temporary, often returned as references
    std::vector<MatrixXd> mats;

    // Very temporary, used for intermediate evaluations
    std::vector<MatrixXd> tmps;

    std::vector<GeneralizedEigenSolver<MatrixXd>> eigs;
};

////////////////////////////////////////////////////////////////////////////////

// Represents a ray as a matrix pencil A + t*B;
struct Pencil {
    MatrixXd mat_A;
    MatrixXd mat_B;
    // To avoid allocation, we work on blocks of the A and B matrix
    size_t rows;
    size_t cols;

    Pencil(size_t rows, size_t cols)
        : mat_A(rows, cols), mat_B(rows, cols), rows(rows), cols(cols)
    {
        // Nothing to do here
    }

    // Performs an in-place reduction, modifying the upper-left corner
    // of A and B and updating rows and cols to reflect the new size.
    bool reduce_step(Scratch& scratch) {
        ENTERING_REAL_TIME_CRITICAL_CODE();

        const auto& A = mat_A.topLeftCorner(rows, cols);
        const auto& B = mat_B.topLeftCorner(rows, cols);

        const auto& svd1 = scratch.svdV(B);
        const auto r = svd1.rank();
        if (r == B.cols()) {
            // B has full column rank, so we're done!
            return false;
        }
        const auto& A_V = scratch.matmul(A, svd1.matrixV());

        // Compute SVD of A1
        const auto& A1 = scratch.rightCols(A_V, B.cols() - r);
        const auto& svd2 = scratch.svdU(A1);
        assert(&svd1 != &svd2);

        const size_t k = svd2.rank();

        const auto& U2T = scratch.transpose(svd2.matrixU());

        const auto& An = scratch.matmul(scratch.matmul(U2T, A), svd1.matrixV());
        const size_t q = An.rows() - k;
        mat_A.topLeftCorner(k, r).noalias() = An.block(An.rows() - k, 0, k, r);

        const auto& Bn = scratch.matmul(scratch.matmul(U2T, B), svd1.matrixV());
        mat_B.topLeftCorner(k, r).noalias() = Bn.block(An.rows() - k, 0, k, r);

        rows = k;
        cols = r;

        EXITING_REAL_TIME_CRITICAL_CODE();
        return true;
    }

    void reduce(Scratch& scratch) {
        while (reduce_step(scratch)) {
            // Keep going
        }
        if (rows != cols) {
            ENTERING_REAL_TIME_CRITICAL_CODE();
            // In-place transpose of active region
            for (unsigned r=0; r < rows; ++r) {
                for (unsigned c=0; c < cols; ++c) {
                    double tmp = mat_A(r, c);
                    mat_A(r, c) = mat_A(c, r);
                    mat_A(c, r) = tmp;

                    tmp = mat_B(r, c);
                    mat_B(r, c) = mat_B(c, r);
                    mat_B(c, r) = tmp;
                }
            }
            const size_t tmp = rows;
            rows = cols;
            cols = tmp;
            EXITING_REAL_TIME_CRITICAL_CODE();
            reduce(scratch);
        }
    }

    void eigenvalues(Scratch& scratch) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        assert(rows == cols);
        auto& A = scratch.mat(rows, rows);
        A = mat_A.topLeftCorner(rows, cols);
        auto& B = scratch.tmp(rows, rows);
        B = mat_B.topLeftCorner(rows, cols);

        auto& solver = scratch.eigs[rows];
        EXITING_REAL_TIME_CRITICAL_CODE();

        // Alas, it's not possible to disable all allocation in the solver
        solver.compute(A, B, false);

        // Issue #2436 documents that these return copies instead of reference
        const auto& alphas = solver.betas();
        const auto& betas = solver.betas();
        for (size_t i=0; i < alphas.size(); ++i) {
            std::cout << alphas[i] << " " << betas[i] << " " << alphas[i] / betas[i] << "\n";
        }
    }

};

////////////////////////////////////////////////////////////////////////////////

struct Mrep {
    MatrixXd M;
    Matrix<double, 3, 2> bbox;
    Vector2i v;

    static Mrep build(const MatrixXv& b) {
        // Pick a v that ensures the drop-of-rank property, based on Section 3.2
        const size_t degree1 = b.rows() - 1;
        const size_t degree2 = b.cols() - 1;
        const Vector2i v(2 * std::min(degree1, degree2) - 1,
                         std::max(degree1, degree2) - 1);

        const auto s = S_v(b, v);
        FullPivLU<MatrixXd> lu_decomp(s);
        MatrixXd M = lu_decomp.kernel();

        // Calculate the bounding box of this patch in XYZ space
        Matrix<double, 3, 2> bbox;
        bbox.col(0) = b(0,0);
        bbox.col(1) = b(0,0);
        for (size_t i=0; i < b.rows(); ++i) {
            for (size_t j=0; j < b.cols(); ++j) {
                bbox.col(0) = bbox.col(0).cwiseMin(b(i, j));
                bbox.col(1) = bbox.col(1).cwiseMax(b(i, j));
            }
        }

        return Mrep { M, bbox, v };
    }

    size_t rows() const {
        return M.rows() / 4;
    }
    size_t cols() const {
        return M.cols();
    }

    // Evaluates the m-rep into the given matrix at a particular position
    void eval(Vector3d pos, MatrixXd& out) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        const size_t r = M.rows() / 4;
        out.noalias() = M.topRows(r);
        out.noalias() += pos.x() * M.middleRows(r, r);
        out.noalias() += pos.y() * M.middleRows(2*r, r);
        out.noalias() += pos.z() * M.bottomRows(r);
        EXITING_REAL_TIME_CRITICAL_CODE();
    }

    // Builds a parameterized ray as a matrix pencil A + t*B
    void ray(Vector3d ray_origin, Vector3d ray_dir, Pencil& r) const
    {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        eval(ray_origin, r.mat_A);
        ray_origin.noalias() += ray_dir;
        eval(ray_origin, r.mat_B);
        r.mat_B -= r.mat_A;
        r.rows = rows();
        r.cols = cols();
        EXITING_REAL_TIME_CRITICAL_CODE();
    }

    Vector2d preimages(Vector3d pos, Scratch& scratch) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        auto& m = scratch.mat(rows(), cols());
        eval(pos, m);
        const auto& svd = scratch.svdU(m);
        const auto n = svd.matrixU().rightCols<1>();

        const size_t h = n.rows() / (v[1] + 1) * 2;
        auto& A = scratch.mat(h, 1);
        auto& B = scratch.tmp(h, 1);

        const auto stride = v[1] + 1;
        for (size_t i=0; i < h/2; ++i) {
            const size_t offset = i * stride;
            A(i, 0) = v[1] * n[offset] + n[1 + offset];
            A(i + h/2, 0) = v[1] * n[v[1] + offset] + n[v[1] - 1 + offset];
            B(i, 0) = n[1 + offset];
            B(i + h/2, 0) = v[1] * n[v[1] + offset];
        }
        const double x = scratch.solve(A, B);

        const size_t offset = v[0] + 1;
        for (size_t i=0; i < h/2; ++i) {
            A(i, 0) = v[0] * n[i] + n[i + offset];
            A(i + h/2, 0) = v[0] * n[n.rows() - offset + i - 1] + n[n.rows() - 2*offset + i - 1];
            B(i, 0) = n[i + offset];
            B(i + h/2, 0) = v[0] * n[n.rows() - offset + i - 1];
        }
        const double y = scratch.solve(A, B);

        EXITING_REAL_TIME_CRITICAL_CODE();
        return Vector2d{x, y};
    }
};

////////////////////////////////////////////////////////////////////////////////

int main() {
    init();

    std::ifstream file("../../teapot.bpt");
    if (!file) {
        std::cerr << "Could not load teapot";
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const auto patches = parse_bpt(buffer);

    auto mrep = Mrep::build(patches[0]);

    auto pt = patches[0](0,0);
    const auto dir = Vector3d{0,0,1};
    pt -= 2.5 * dir;

    auto ray = Pencil(mrep.rows(), mrep.cols());
    mrep.ray(pt, dir, ray);

    Scratch scratch(mrep.rows(), mrep.cols());

    ray.reduce(scratch);
    ray.eigenvalues(scratch);

    std::cout << mrep.preimages(patches[0](0,0), scratch).transpose() << "\n";
    std::cout << mrep.preimages(patches[0](0,3), scratch).transpose() << "\n";
    std::cout << mrep.preimages(patches[0](3,0), scratch).transpose() << "\n";
    std::cout << mrep.preimages(patches[0](3,3), scratch).transpose() << "\n";
}
