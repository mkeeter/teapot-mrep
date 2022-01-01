#include <stdio.h>
#include <stdbool.h>

bool ray_box(const double* const ray_origin,
             const double* const ray_dir,
             const double* const box_min,
             const double* const box_max,
             double* best_dist) {
    bool any_hit = false;
    for (int axis=0; axis < 3; ++axis) {
        if (ray_dir[axis] != 0.0) {
            const double dmin = (box_min[axis] - ray_origin[axis]) / ray_dir[axis];
            const double dmax = (box_max[axis] - ray_origin[axis]) / ray_dir[axis];
            const double ds[2] = {dmin, dmax};
            for (int i=0; i < 2; ++i) {
                const double d = ds[i];
                if (d >= 0 && (any_hit == 0 || d < *best_dist)) {
                    bool valid = true;
                    for (int j=0; j < 3; ++j) {
                        if (j == axis) {
                            continue;
                        }
                        const double p = ray_origin[j] + d * ray_dir[j];
                        valid &= p >= box_min[j];
                        valid &= p <= box_max[j];
                    }
                    if (valid) {
                        *best_dist = d;
                        any_hit = true;
                    }
                }
            }
        }
    }
    return any_hit;
}
