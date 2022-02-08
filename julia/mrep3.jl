using LinearAlgebra
using Plots
using Images

" Calculates the values of a Bernstein polynomial at `u` "
function bernstein(i, degree, u)
    binomial(degree, i) .* u.^i .* (1 .- u).^(degree - i)
end

" Calculates the derivatives of a Bernstein polynomial at `u` "
function bernstein_deriv(i, degree, u)
    a = if i != 0
        i .* u.^(i - 1) .* (1 .- u).^(degree - i)
    else
        zeros(size(u))
    end
    b = if degree - i != 0
        (i - degree) .* u.^i .* (1 .- u).^(degree - i - 1)
    else
        zeros(size(u))
    end
    binomial(degree, i) * (a + b)
end

" Samples the Bézier surface `b` at points `u × v` "
function surface_points(b, u, v)
    degree1 = size(b)[1] - 1
    degree2 = size(b)[2] - 1
    dimension = size(b)[3]
    out = zeros((length(u), length(v), dimension))
    for i in 0:degree1
        for j in 0:degree2
            wu = bernstein(i, degree1, u)
            wv = bernstein(j, degree2, v)
            w = wu * transpose(wv)
            for k in 1:dimension
                out[:,:,k] += b[i + 1, j + 1, k] .* w
            end
        end
    end
    out
end

" Finds the derivatives of a Bézier surface b at points u × v, returning an
  array of size (u, v, dim)"
function surface_derivs(b, u, v)
    degree1 = size(b)[1] - 1
    degree2 = size(b)[2] - 1
    dimension = size(b)[3]

    d_du = zeros((length(u), length(v), dimension))
    d_dv = zeros((length(u), length(v), dimension))
    out = Array{Float64}(undef, (length(u), length(v), dimension))
    for i in 0:degree1
        for j in 0:degree2
            dw_du = bernstein_deriv(i, degree1, u)
            wv = bernstein(j, degree2, v)
            dws_du = dw_du * transpose(wv)

            wu = bernstein(i, degree1, u)
            dw_dv = bernstein(j, degree2, v)
            dws_dv = wu * transpose(dw_dv)

            for k in 1:dimension
                d_du[:,:,k] += b[i + 1, j + 1, k] .* dws_du
                d_dv[:,:,k] += b[i + 1, j + 1, k] .* dws_dv
            end
        end
    end
    for i in 1:length(u)
        for j in 1:length(v)
            out[i, j, :] = normalize(cross(d_du[i, j, :], d_dv[i, j, :]))
        end
    end
    out
end

" Reads a .bpt file from disk and returns a set of patches "
function read_bpt(filename)
    out = Any[]
    open(filename) do io
        patch_count = parse(Int64, readline(io))
        for p in 1:patch_count
            line = readline(io)
            size = Tuple(parse.(Int64, split(line, ' ')))
            patch = zeros((size[1] + 1, size[2] + 1, 3))
            for i in 0:size[1]
                for j in 0:size[2]
                    line = readline(io)
                    pt = parse.(Float64, split(line, ' '))
                    patch[i + 1, j + 1, :] = pt
                end
            end
            push!(out, patch)
        end
    end
    out
end

" Builds the i + j weight of B^d_i * B^v_j in the B^{d + v} basis
  d, i, and j can be higher-dimensional for product surfaces"
function base_weight(b, d, v, i)
    prod(binomial.(d, i) .* binomial.(v, j) ./ binomial(d + v, i + j) .* b[i])
end

" Builds the S_v matrix for a Bézier surface, using the equation on p. 4 "
function S_v(b)
    degree1 = size(b)[1] - 1
    degree2 = size(b)[2] - 1
    dimension = size(b)[3]
    # Pick a v that ensures the drop-of-rank property, based on Section 3.2
    v = (2 * min(degree1, degree2) - 1, max(degree1, degree2) - 1)

    stride = (v[1] + 1) * (v[2] + 1)
    out = zeros(((degree1 + v[1] + 1) * (degree2 + v[2] + 1), 4 * stride))
    for axis in 0:dimension
        c = if axis == 0
            ones(size(b)[1:2])
        else
            b[:, :, axis]
        end
        for k in 0:v[1]
            v_k = binomial(v[1], k)
            for i in 0:degree1
                mul_ik = v_k * binomial(degree1, i) / binomial(v[1] + degree1, i + k)
                for l in 0:v[2]
                    v_l = binomial(v[2], l)
                    col = l + k * (v[2] + 1) + axis * stride
                    for j in 0:degree1
                        row = (j + l) + (i + k) * (v[2] + degree2 + 1)
                        mul_jl = v_l * binomial(degree2, j) / binomial(v[2] + degree2, j + l)
                        out[row + 1, col + 1] +=
                            mul_ik * mul_jl * c[i + 1, j + 1]
                    end
                end
            end
        end
    end
    out, v
end

struct Mrep
    m0::Matrix{Float64}
    mx::Matrix{Float64}
    my::Matrix{Float64}
    mz::Matrix{Float64}
    bbox::Array{Tuple{Float64, Float64}, 3}
    v::Tuple{Int64, Int64}
end

function build_mrep(b)
    s, v = S_v(b)
    null = nullspace(s)
    i = size(null)[1] ÷ 4
    Mrep(null[1:i,:],
         null[(i + 1):2*i, :],
         null[(2*i + 1):3*i, :],
         null[(3*i + 1):4*i, :],
         extrema(b, dims=(1,2)), v)
end

function eval_mrep(m::Mrep, x::Float64, y::Float64, z::Float64)
    m.m0 + x * m.mx + y * m.my + z * m.mz
end

" Checks the given ray against the bounding box of the given m-rep, returning
  either a minimum distance (if there's a possible hit) or false
"
function min_distance_bbox(m::Mrep, ray_origin, ray_dir)
    any_hit = false
    best_dist = 0.0
    for axis in 1:3
        if ray_dir[axis] ≈ 0
            continue
        end
        for d in (m.bbox[axis] .- ray_origin[axis]) ./ ray_dir[axis]
            if d >= 0 && (any_hit == 0 || d < best_dist)
                valid = true
                for j in 1:3
                    if j == axis
                        continue
                    end
                    p = ray_origin[j] + d * ray_dir[j]
                    valid &= p >= m.bbox[j][1]
                    valid &= p <= m.bbox[j][2]
                end
                if valid
                    best_dist = d
                    any_hit = true
                end
            end
        end
    end
    if any_hit
        best_dist
    else
        false
    end
end

" Parameterizes a ray R(t) = o + td, returning a pair of matrices A, B such
  that M(R(t)) = A - tB"
function parameterize_ray(M, o, d)
    A = eval_mrep(M, o...)
    B = eval_mrep(M, (o + d)...) - A
    (A, B)
end

" Performs a single step of matrix pencil reduction, based on Section 2.3 in
  'A Line/Trimmed NURBS Surface Intersection Algorithm Using Matrix
  Representations'
"
function reduce_pencil_step(A, B)
    s1 = svd(B, full=true)

    # Check to see whether B is full rank, using the standard rtol
    rtol = min(size(B)...) * eps(eltype(B))
    r = sum(s1.S .> (s1.S[1] * rtol))
    if ==(size(B)...) ||  r == size(B)[2]
        return A, B
    end

    BV1 = B * s1.V
    AV1 = A * s1.V
    A1 = AV1[:, r + 1:end]
    s2 = svd(A1, full=true)
    k = size(s2.S)[1]
    A = transpose(s2.U) * A * s1.V
    B = transpose(s2.U) * B * s1.V
    A[end-k + 1:end, 1:r], B[end-k + 1:end, 1:r]
end

" Reduces the matrix pencil A + tB until it reaches full rank or disappears"
function reduce_pencil(A, B)
    while true
        A_, B_ = reduce_pencil_step(A, B)
        if size(A_) == size(A)
            break
        end
        A, B = A_, B_
    end
    if ==(size(A)...)
        A, B
    else
        reduce_pencil(transpose(A), transpose(B))
    end
end

" Returns eigenvalues of the given matrix pencil.  We attempt to use matrix
  pencil reduction, but fall back to a higher-power algorithm if that fails.
"
function pencil_eigenvalues(A, B)
    A_, B_ = reduce_pencil(A, B)
    if size(A_) != (0, 0)
        eigvals(A_, B_)
    else
        println("oh no")
        []
    end
end

" Calculates the first intersections of the given ray with the m-rep patch M,
return (distance, uv coordinates) on the patch or false "
function ray_hit(M, ray_origin, ray_dir, best_dist)
    A, B = parameterize_ray(M, ray_origin, ray_dir)
    r0 = min(size(A)...)
    eigs = pencil_eigenvalues(A, B)

    rtol = sqrt(eps(eltype(A))) * max(abs.(eigs)...) # ??
    hits = Float64[]
    for e in eigs
        if imag(e) < rtol && real(e) <= 0 && rank(A - real(e) * B) != r0
            push!(hits, -real(e))
        end
    end

    rtol = sqrt(eps(eltype(A))) # ???
    for e in sort(unique(hits))
        # Only check distances which are closer than our best distance
        if best_dist == false || e < best_dist
            pt = ray_origin + e * ray_dir
            x, y = preimages(M, pt)
            # Return the first distance that gives valid UV coordinates
            if x >= -rtol && x <= 1 + rtol && y >= -rtol && y <= 1 + rtol
                return e, clamp(x, 0, 1), clamp(y, 0, 1)
            end
        else
            # Short-circuit if we're farther than the best distance
            return false
        end
    end
    false
end

" Calculates the preimages of a particular point on an Mrep surface "
function preimages(M, pt)
    n = svd(eval_mrep(M, pt...)).U[:, end]
    v = M.v

    # Use least-squares to robustly solve for parameters
    A = vcat(v[2] * n[1:v[2]+1:end] + n[2:v[2]+1:end],
             v[2] * n[v[2] + 1:v[2] + 1:end] + n[v[2]:v[2] + 1:end])
    B = vcat(n[2:v[2] + 1:end], v[2] * n[v[2] + 1:v[2] + 1:end])
    x = A \ B

    A = vcat(v[1] * n[1:v[2] + 1] + n[v[2] + 2:2*(v[2] + 1)],
             v[1] * n[end - v[2]:end] + n[end-2*v[2]-1:end-v[2]-1])
    B = vcat(n[v[2] + 2:2*(v[2] + 1)], v[1] * n[end - v[2]:end])
    y = A \ B

    x, y
end

" Returns an [image_size x image_size x 3] array of ray starting positions "
function camera_rays(camera_pos, camera_look, camera_scale, image_size)
    camera_dir = normalize(camera_look - camera_pos)
    camera_up = [0, 0, 1]
    camera_x = normalize(cross(camera_dir, camera_up))
    camera_y = cross(camera_x, camera_dir)
    p = collect(1:image_size) ./ image_size .- 0.5

    r = Array{Float64}(undef, (image_size, image_size, 3))
    for i in 1:image_size
        for j in 1:image_size
            r[j,i,:] = camera_pos + camera_scale *
                       (camera_x * p[i] + camera_y * p[j])
        end
    end
    (r, camera_dir)
end

function raytrace(mreps, ray_origin, ray_dir)
    todo = Tuple{Float64, Int64}[]
    for (i, m) in enumerate(mreps)
        d = min_distance_bbox(m, ray_origin, ray_dir)
        if d != false
            push!(todo, (d, i))
        end
    end
    sort!(todo)

    best_dist = false
    best_uv = undef
    best_index = undef
    for (d, i) in todo
        # Skip cases where the closest hit is already farther than our target
        if best_dist != false && d >= best_dist
            break
        end
        hit = ray_hit(mreps[i], ray_origin, ray_dir, best_dist)
        if hit != false
            best_dist = hit[1]
            best_uv = (hit[2], hit[3])
            best_index = i
        end
    end
    if best_dist != false
        best_dist, best_index, best_uv
    else
        false
    end
end

function render(mreps, camera_pos, camera_look, image_size)
    (rays, ray_dir) = camera_rays(camera_pos, camera_look, 6, image_size)
    img = zeros(image_size, image_size)
    for row in 1:image_size
        println(row)
        for col in 1:image_size
            ray_pos = rays[row,col,:]
            hit = raytrace(mreps, ray_pos, ray_dir)
            if hit != false
                img[row,col] = hit[1]
            end
        end
    end
    img
end

patches = read_bpt("teapot.bpt")
mreps = map(build_mrep, patches)
camera_pos = [5,5,5]
camera_look = [0.07,0.1,1]
#img = render(mreps, camera_pos, camera_look, 400)
