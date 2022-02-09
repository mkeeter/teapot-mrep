#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
static double N_CHOOSE_K[MAX_N_CHOOSE_K][MAX_N_CHOOSE_K] = {{0.0}};

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
        for (size_t j=0; j < n + 1; ++j) {
            for (size_t k=0; k < m + 1; ++k) {
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
        for (int k=0; k < v[0] + 1; ++k) {
            const auto v_k = N_CHOOSE_K[v[0]][k];
            for (int l=0; l < v[1] + 1; ++l) {
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

struct Scratch; // Forward declaration

// Represents a ray as a matrix pencil A + t*B;
struct Pencil {
    MatrixXd mat_A;
    MatrixXd mat_B;

    // To avoid allocation, we work on blocks of the A and B matrix
    size_t rows;
    size_t cols;

    // Allocates enough space for a size x size pencil, but doesn't lock
    // it in (by leaving rows and cols at 0)
    Pencil(size_t size)
        : mat_A(size, size), mat_B(size, size), rows(0), cols(0)
    {
        // Nothing to do here
    }

    // Defined below after struct Scratch
    bool reduce_step(Scratch& scratch);
    void reduce(Scratch& scratch);
    MatrixXd& eigenvalues(Scratch& scratch) const;
};

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    Scratch(size_t size)
        : stride(size + 1), pencil(stride)
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

    // SVD solvers for various sides, indexed as rows + cols * stride
    std::vector<JacobiSVD<MatrixXd>> svdsU;
    std::vector<JacobiSVD<MatrixXd>> svdsV;

    // SVD solver for single-column matrices (used for least-squares),
    // indexed as 0-stride
    std::vector<JacobiSVD<MatrixXd>> svdsUV;

    // Somewhat temporary, often returned as references,
    // indexed as rows + cols * stride
    std::vector<MatrixXd> mats;

    // Very temporary, used for intermediate evaluations,
    // indexed as rows + cols * stride
    std::vector<MatrixXd> tmps;

    // Array of eigenvalue solvers, indexed as 0-stride
    std::vector<GeneralizedEigenSolver<MatrixXd>> eigs;

    Pencil pencil;
};

////////////////////////////////////////////////////////////////////////////////

// Performs an in-place reduction, modifying the upper-left corner
// of A and B and updating rows and cols to reflect the new size.
bool Pencil::reduce_step(Scratch& scratch) {
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
    mat_A.topLeftCorner(k, r).noalias() = An.block(An.rows() - k, 0, k, r);

    const auto& Bn = scratch.matmul(scratch.matmul(U2T, B), svd1.matrixV());
    mat_B.topLeftCorner(k, r).noalias() = Bn.block(An.rows() - k, 0, k, r);

    rows = k;
    cols = r;

    EXITING_REAL_TIME_CRITICAL_CODE();
    return true;
}

void Pencil::reduce(Scratch& scratch) {
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

// Returns real (or almost-real) positive eigenvalues
MatrixXd& Pencil::eigenvalues(Scratch& scratch) const {
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
    const auto& alphas = solver.alphas();
    const auto& betas = solver.betas();

    size_t count = 0;
    for (long i=0; i < alphas.size(); ++i) {
        // XXX: epsilon?
        if (alphas[i].imag() == 0.0) {
            count++;
        }
    }

    auto& out = scratch.mat(count, 1);
    count = 0;
    for (long i=0; i < alphas.size(); ++i) {
        if (alphas[i].imag() == 0.0) {
            out(count++, 0) = -alphas[i].real() / betas[i];
        }
    }
    return out;
}

////////////////////////////////////////////////////////////////////////////////

struct Hit {
    bool valid = false;
    double distance;
    size_t index;
    Vector2d uv;
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
        for (long i=0; i < b.rows(); ++i) {
            for (long j=0; j < b.cols(); ++j) {
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
    template <typename T>
    void eval(Vector3d pos, T out) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        const size_t r = M.rows() / 4;
        out.noalias() = M.topRows(r);
        out.noalias() += pos.x() * M.middleRows(r, r);
        out.noalias() += pos.y() * M.middleRows(2*r, r);
        out.noalias() += pos.z() * M.bottomRows(r);
        EXITING_REAL_TIME_CRITICAL_CODE();
    }

    // Builds a parameterized ray as a matrix pencil A + t*B
    void ray(Vector3d ray_origin, Vector3d ray_dir, Pencil& p) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        const size_t r = rows();
        const size_t c = cols();
        eval(ray_origin, p.mat_A.topLeftCorner(r, c));
        ray_origin.noalias() += ray_dir;
        eval(ray_origin, p.mat_B.topLeftCorner(r, c));
        p.mat_B.topLeftCorner(r, c) -= p.mat_A.topLeftCorner(r, c);
        p.rows = r;
        p.cols = c;
        EXITING_REAL_TIME_CRITICAL_CODE();
    }

    // Calculates the minimum distance from the given ray to the bounding
    // box of this m-rep, or a hit with valid = false if there is no hit.
    Hit min_distance(Vector3d ray_origin, Vector3d ray_dir) const {
        Hit out;
        for (size_t axis=0; axis < 3; ++axis) {
            if (ray_dir[axis] == 0.0) {
                // TODO: epsilon?
                continue;
            }
            // Check against min and max bounding box sides
            for (size_t i=0; i < 2; ++i) {
                const double d = (bbox(axis, i) - ray_origin[axis]) / ray_dir[axis];
                if (d >= 0 && (!out.valid || d < out.distance)) {
                    bool valid = true;
                    for (size_t j=0; j < 3; ++j) {
                        if (i == j) {
                            continue;
                        }
                        const double p = ray_origin[j] + d * ray_dir[j];
                        valid &= (p >= bbox(j, 0)) & (p <= bbox(j, 1));
                    }
                    if (valid) {
                        out.distance = d;
                        out.valid = true;
                    }
                }
            }
        }
        return out;
    }

    Vector2d preimages(Vector3d pos, Scratch& scratch) const {
        ENTERING_REAL_TIME_CRITICAL_CODE();
        auto& m = scratch.mat(rows(), cols());
        eval<MatrixXd&>(pos, m);
        const auto& svd = scratch.svdU(m);
        const auto n = svd.matrixU().rightCols<1>();

        const size_t h = n.rows() / (v[1] + 1) * 2;
        auto& A = scratch.mat(h, 1);
        auto& B = scratch.tmp(h, 1);

        const auto stride = v[1] + 1;
        for (size_t i=0; i < h/2; ++i) {
            const size_t offset = i * stride;
            const size_t j = i + h/2;
            B(i, 0) = n[1 + offset];
            A(i, 0) = v[1] * n[offset] + B(i, 0);
            B(j, 0) = v[1] * n[v[1] + offset];
            A(j, 0) = B(j, 0) + n[v[1] - 1 + offset];
        }
        const double x = scratch.solve(A, B);

        const size_t offset = v[0] + 1;
        for (size_t i=0; i < h/2; ++i) {
            const size_t j = i + h/2;
            B(i, 0) = n[i + offset];
            A(i, 0) = v[0] * n[i] + B(i, 0);
            B(j, 0) = v[0] * n[n.rows() - offset + i - 1];
            A(j, 0) = B(j, 0) + n[n.rows() - 2*offset + i - 1];
        }
        const double y = scratch.solve(A, B);

        EXITING_REAL_TIME_CRITICAL_CODE();
        return Vector2d{x, y};
    }
};

////////////////////////////////////////////////////////////////////////////////

Hit raytrace(Vector3d ray_origin, Vector3d ray_dir,
             const std::vector<Mrep>& mreps,
             Scratch& scratch)
{
    // Sort by minimum distance, skipping invalid options
    std::vector<std::tuple<double, size_t>> todo;
    size_t index = 0;
    for (const auto& m : mreps) {
        const auto h = m.min_distance(ray_origin, ray_dir);
        todo.push_back(std::make_tuple(h.valid ? h.distance : -1.0, index++));
    }

    std::sort(todo.begin(), todo.end());

    // If every ray doesn't hit, then return immediately
    Hit hit;
    if (std::get<0>(todo[todo.size() - 1]) == -1.0) {
        return hit;
    }

    for (auto& t : todo) {
        const auto min_distance = std::get<0>(t);
        if (min_distance == -1.0) {
            // Skip non-hitting rays
            continue;
        } else if (hit.valid && min_distance >= hit.distance) {
            // We're done if all future options have a farther min distance
            break;
        }
        const auto index = std::get<1>(t);
        const auto& mrep = mreps[index];

        mrep.ray(ray_origin, ray_dir, scratch.pencil);
        scratch.pencil.reduce(scratch);

        const auto& eigs = scratch.pencil.eigenvalues(scratch);
        for (int i=0; i < eigs.rows(); ++i) {
            const double d = eigs(i, 0);
            if (d <= 0.0 || (hit.valid && d >= hit.distance)) {
                continue;
            }
            // Check that the collision is inside this patch's bounding box,
            // before doing the expensive preimage computation
            bool inside_bbox = true;
            const Vector3d pt = ray_origin + d * ray_dir;
            for (size_t axis=0; axis < 3; ++axis) {
                inside_bbox &= (pt[axis] >= mrep.bbox(axis, 0)) &
                               (pt[axis] <= mrep.bbox(axis, 1));
            }
            if (!inside_bbox) {
                continue;
            }

            const Vector2d uv = mrep.preimages(pt, scratch);
            if (uv.x() < 0.0 || uv.x() > 1.0 || uv.y() < 0.0 || uv.y() > 1.0) {
                continue;
            }

            hit.valid = true;
            hit.index = index;
            hit.distance = d;
            hit.uv = uv;
        }
    }

    return hit;
}

MatrixXv render(const std::vector<Mrep>& mreps,
                Vector3d camera_pos,
                Vector3d camera_look,
                double camera_scale,
                size_t image_size)
{
    // Find a maximum bounding size for our scratch data
    size_t s = 0;
    for (const auto& m: mreps) {
        s = std::max(s, std::max(m.rows(), m.cols()));
    }
    Scratch scratch(s);

    const Vector3d camera_dir = (camera_look - camera_pos).normalized();
    const Vector3d camera_up{0.0, 0.0, 1.0};
    const Vector3d camera_x = camera_dir.cross(camera_up).normalized();
    const Vector3d camera_y = camera_x.cross(camera_dir);

    MatrixXv out(image_size, image_size);
    for (size_t i=0; i < image_size; ++i) {
        std::cout << i << "/" << image_size << std::endl;
        const Vector3d row_pos = camera_pos +
                (i / double(image_size) - 0.5) * camera_scale * camera_x;
        for (size_t j=0; j < image_size; ++j) {
            const Vector3d ray_pos = row_pos +
                (j / double(image_size) - 0.5) * camera_scale * camera_y;
            const auto h = raytrace(ray_pos, camera_dir, mreps, scratch);
            if (h.valid) {
                out(i, j) = Vector3d(1.0, 0.0, 0.0);
            } else {
                out(i, j) = Vector3d(0.0, 0.0, 0.0);
            }
        }
    }

    return out;
}

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

    std::vector<Mrep> mreps;
    for (const auto& p : patches) {
        mreps.push_back(Mrep::build(p));
    }
    const Vector3d camera_pos{3, 3, 3};
    const Vector3d camera_look{0.07, 0.1, 1.4};
    const double camera_scale = 6;
    const size_t image_size = 100;
    const auto img = render(mreps, camera_pos, camera_look, camera_scale, image_size);
    uint32_t* data = new uint32_t[image_size * image_size];
    for (size_t i=0; i < image_size; ++i) {
        for (size_t j=0; j < image_size; ++j) {
            const auto rgb = img(j, image_size - i - 1);
            if (rgb.norm() != 0.0) {
                const uint8_t r = fabs(rgb.x()) * 255;
                const uint8_t g = fabs(rgb.y()) * 255;
                const uint8_t b = fabs(rgb.z()) * 255;
                data[i * image_size + j] =
                    ((uint32_t)r << 0) |
                    ((uint32_t)g << 8) |
                    ((uint32_t)b << 16)|
                    (0xFF << 24);
            } else {
                data[i * image_size + j] = 0;
            }
        }
    }
    stbi_write_png("out.png", image_size, image_size, 4, data, image_size * 4);
    delete [] data;
}
