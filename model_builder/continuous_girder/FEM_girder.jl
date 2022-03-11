#=
TODO/IMPROVE:
    * enforce the same size of some input arguments using parametric types =#

include(abspath(".\\continuous_girder\\FEM_utils.jl"))
include(abspath(".\\continuous_girder\\utils.jl"))

"""
    fem_prismatic_twin_girder(args...; kwargs...) -> (Ks, nodes, idx_keep, lin_idx_springs)

Create a 2D FEM for a continuous twin-girder from `spans` and `flexural_stiffnesses`.
Loads are not considered.

# Arguments

- `spans::AbstractArray{Float64, 1}`: Span lengths of the girder.
- `flexural_stiffnesses`

# Keywords

- `support_dofs`: Boundary conditions at the ends of each span.
    Row `i` defines the dofs of support `i` from left to right:
    `[y_translation, z_rotation]`. By default only `y_translation` is fixed (`0`).
- `num_elem_per_span`: a vector with elements as the number of elements per span.
- `additional_node_positions`: Use to place nodes at positions needed later, e.g. load positions and/or sensor positions.
- `left_hinge_positions`: a vector of internal hinge positions for the left girder.
- `right_hinge_positions`: a vector of internal hinge positions for the right girder.

# Returns

- `Ks`: global stiffness matrix with support boundary conditions applied.
- `nodes`:
- `idx_keep`: `Ks = K[idx_keep, idx_keep]`
- `lin_idx_springs`: a dictionary with the linear indices of the spring stiffness positions
    in the `Ks` matrix. Indices for the diagonal and off diagonal elements of the local
    spring stiffness matrices are given separately (two fields in the Dict).

Node numbering convention to create a narrow global stiffness matrix:

^y
|
o--->x (longitudinal axis of the girder)

1    3    5     2*ii-1
o====o====o ... o ...   left (1st) girder
o====o====o ... o ...   right (2nd) girder
2    4    6     2*ii

Considering the stiffness mx/or displacement vector with all dofs (without removing some
due to constrained dofs):
* first girder translations: 1:4:end
* first girder rotations: 2:4:end
* second girder translations: 3:4:end
* second girder rotations: 4:4:end
"""
function fem_prismatic_twin_girder(
    spans::AbstractArray{Float64,1},
    flexural_stiffnesses::AbstractArray{Float64,1};
    support_dofs::AbstractArray{Int64}=zeros(Int, 0, 2),
    num_elem_per_span::AbstractArray{Int64,1}=zeros(Int, 0),
    additional_node_positions::AbstractArray{Float64,1}=zeros(0),
    spring_positions::AbstractArray{Float64,1}=zeros(0),
    spring_stiffnesses::AbstractArray{Float64,1}=zeros(0),
    left_hinge_positions::AbstractArray{Float64,1}=zeros(0),
    right_hinge_positions::AbstractArray{Float64,1}=zeros(0),
)
    # ................................................
    # Initialize
    # ................................................
    support_positions = [0, cumsum(spans)...]

    if length(support_dofs) == 0
        support_dofs = repeat([1, 0]', length(spans) + 1)
    end

    if length(num_elem_per_span) == 0
        num_elem_per_span = fill(6, length(spans))
    end

    append!(additional_node_positions, spring_positions)

    # to keep the nodes of the two girders symmetric
    append!(additional_node_positions, left_hinge_positions)
    append!(additional_node_positions, right_hinge_positions)

    # ................................................
    # Assemble the stiffness mx for each girder (g)
    # without boundary conditions
    # ................................................
    K_g_left, nodes_g, idx_keep_g = fem_prismatic_single_girder(
        spans, flexural_stiffnesses;
        support_dofs=repeat([0, 0]', length(spans) + 1),
        num_elem_per_span=num_elem_per_span,
        additional_node_positions=additional_node_positions,
        hinge_positions=left_hinge_positions,
    )

    K_g_right, _, _ = fem_prismatic_single_girder(
        spans, flexural_stiffnesses;
        support_dofs=repeat([0, 0]', length(spans) + 1),
        num_elem_per_span=num_elem_per_span,
        additional_node_positions=additional_node_positions,
        hinge_positions=right_hinge_positions,
    )

    # ................................................
    # Assemble the stiffness mx for the twin-girder (t)
    # with boundary conditions
    # ................................................
    Ks_t, nodes_t, idx_keep, lin_idx_springs = _fem_twin_girder(
        K_g_left,
        K_g_right,
        nodes_g,
        spring_positions,
        spring_stiffnesses,
        support_dofs,
        support_positions,
    )
    return Ks_t, nodes_t, idx_keep, lin_idx_springs
end

"""
This function relates to `fem_prismatic_twin_girder` the same way as
`fem_general_single_girder` relates to `fem_prismatic_single_girder`.
"""
function fem_general_twin_girder(
    node_xs_g::AbstractArray{Float64,1},
    elem_EI_g::AbstractArray{Float64,1},
    support_positions::AbstractArray{Float64,1};
    support_dofs::AbstractArray{Int64}=zeros(Int, 0, 2),
    spring_positions::AbstractArray{Float64,1}=zeros(0),
    spring_stiffnesses::AbstractArray{Float64,1}=zeros(0),
    left_hinge_positions::AbstractArray{Float64,1}=zeros(0),
    right_hinge_positions::AbstractArray{Float64,1}=zeros(0),
)
    # ................................................
    # Initialize
    # ................................................
    num_node_g = length(node_xs_g)
    num_elem_g = length(elem_EI_g)
    num_dof_g = num_node_g * 2
    num_support = length(support_positions)

    # TODO: to be improved, done twice
    support_dofs = _fem_general_input_check(
        node_xs_g, elem_EI_g, support_positions, support_dofs
    )

    # ................................................
    # Assemble the stiffness mx for each girder (g)
    # without boundary conditions
    # ................................................
    K_g_left, nodes_g, idx_keep_g = fem_general_single_girder(
        node_xs_g, elem_EI_g, support_positions;
        support_dofs=repeat([0, 0]', num_support),
        hinge_positions=left_hinge_positions,
    )

    K_g_right, _, _ = fem_general_single_girder(
        node_xs_g, elem_EI_g, support_positions;
        support_dofs=repeat([0, 0]', num_support),
        hinge_positions=right_hinge_positions,
    )

    # ................................................
    # Assemble the stiffness mx for the twin-girder (t)
    # with boundary conditions
    # ................................................
    Ks_t, nodes_t, idx_keep, lin_idx_springs = _fem_twin_girder(
        K_g_left,
        K_g_right,
        nodes_g,
        spring_positions,
        spring_stiffnesses,
        support_dofs,
        support_positions,
    )
    return Ks_t, nodes_t, idx_keep, lin_idx_springs
end


function _fem_twin_girder(
    K_g_left,
    K_g_right,
    nodes_g,
    spring_positions,
    spring_stiffnesses,
    support_dofs,
    support_positions,
)
    # ................................................
    # Initialize
    # ................................................
    num_dof_g = size(K_g_left, 1)
    num_node_g = size(nodes_g, 1)
    num_elem_g = size(nodes_g, 1) - 1
    num_dof_t = 2 * num_dof_g

    # ................................................
    # Assemble the stiffness mx for the twin-girder (t)
    # without boundary conditions
    # ................................................

    # filling in the stiffness values of one girder for the upper triangular
    # (+ a few in the lower diagonal that will be ignored when mirroring the mx)
    function fill_in_girder(K_g, girder)
        K_tg = zeros(Float64, num_dof_t, num_dof_t)
        step_t = 4
        step_g = 2
        start_idx_grs = [1, 1]
        start_idx_gcs = [1, 1 + step_g]
        num_blocks = [num_node_g, num_node_g - 1]
        if girder == "left"
            start_idx_trs = [1, 1]
            start_idx_tcs = [1, 1 + step_t]
        elseif girder == "right"
            start_idx_trs = [1 + step_g, 1 + step_g]
            start_idx_tcs = [1 + step_g, 1 + step_t  + step_g]
        else
            throw(ArgumentError)
        end

        for ii in 1:length(num_blocks)
            start_idx_gr = start_idx_grs[ii]
            start_idx_gc = start_idx_gcs[ii]
            start_idx_tr = start_idx_trs[ii]
            start_idx_tc = start_idx_tcs[ii]
            num_block = num_blocks[ii]

            idx_gr = range(start_idx_gr, step=step_g, length=num_block) |> collect
            idx_gc = range(start_idx_gc, step=step_g, length=num_block) |> collect
            idx_tr = range(start_idx_tr, step=step_t, length=num_block) |> collect
            idx_tc = range(start_idx_tc, step=step_t, length=num_block) |> collect

            # diagonal 11
            lin_idx_g = linear_index(idx_gr, idx_gc, num_dof_g)
            lin_idx_t = linear_index(idx_tr, idx_tc, num_dof_t)
            K_tg[lin_idx_t] = K_g[lin_idx_g]

            # diagonal 22
            lin_idx_g = linear_index(idx_gr .+ 1, idx_gc .+ 1, num_dof_g)
            lin_idx_t = linear_index(idx_tr .+ 1, idx_tc .+ 1, num_dof_t)
            K_tg[lin_idx_t] = K_g[lin_idx_g]

            # off diagonal 12
            lin_idx_g = linear_index(idx_gr, idx_gc .+ 1, num_dof_g)
            lin_idx_t = linear_index(idx_tr, idx_tc .+ 1, num_dof_t)
            K_tg[lin_idx_t] = K_g[lin_idx_g]

            # off diagonal 21
            lin_idx_g = linear_index(idx_gr .+ 1, idx_gc, num_dof_g)
            lin_idx_t = linear_index(idx_tr .+ 1, idx_tc, num_dof_t)
            K_tg[lin_idx_t] = K_g[lin_idx_g]
        end
        return K_tg
    end

    K_tg_left = fill_in_girder(K_g_left, "left")
    K_tg_right = fill_in_girder(K_g_right, "right")
    K_t = K_tg_left + K_tg_right

    # make it into a symmetric matrix;
    # while avoid doubling the already filled in lower triangular elements
    K_t = triu(K_t) + tril(K_t', -1)

    # ................................................
    # Nodes
    # ................................................
    nodes_t = Array{Float64}(undef, 2 * num_node_g, 3)
    nodes_t[1:2:end, :] = nodes_g
    nodes_t[2:2:end, :] = nodes_g

    # ................................................
    # Elements
    # ................................................
    elem_nodes_g = _element_nodes(num_node_g) .* 2 .- 1
    elem_nodes_t = Array{Float64}(undef, 2 * num_elem_g, 2)
    elem_nodes_t[1:2:end, :] = elem_nodes_g
    elem_nodes_t[2:2:end, :] = elem_nodes_g .+ 1

    # ................................................
    # Between girder springs
    # ................................................
    if length(spring_positions) > 0
        # add the stiffness values to the global stiffness matrix
        node_pairs = _spring_nodes(nodes_t, spring_positions)
        spring_stiff2 = repeat(spring_stiffnesses, 2)
        lin_idx_off_diag, lin_idx_diag = _linear_index_spring(
            node_pairs, num_dof_t; type="trans"
        )
        K_t[lin_idx_off_diag] -= spring_stiff2
        K_t[lin_idx_diag] += spring_stiff2
    else
        lin_idx_springs = zeros(0)
    end

    # ................................................
    # Apply the boundary conditions to the twin-girder (t)
    # ................................................
    Ks_t, idx_keep = _apply_bcs(K_t, nodes_t, support_positions, support_dofs)

    # get linear indices for the stiffness matrix with boundary conditions applied
    if length(spring_positions) > 0
        K_dummy = zeros(Int8, num_dof_t, num_dof_t)
        K_dummy[lin_idx_off_diag] .= -1
        K_dummy[lin_idx_diag] .= 1
        Ks_dummy = K_dummy[idx_keep, idx_keep]
        lin_off_diag = findall(Ks_dummy[:] .== -1)
        lin_diag = findall(Ks_dummy[:] .== 1)

        lin_idx_springs = Dict("off_diag" => lin_off_diag, "diag" => lin_diag)
    end

    return Ks_t, nodes_t, idx_keep, lin_idx_springs
end


"""
    fem_prismatic_single_girder(args...; kwargs...) -> (Ks, nodes, idx_keep)

Create a 2D FEM for a continuous girder from `spans` and `flexural_stiffnesses`. Loads are
not considered. Returns the global stiffness matrix (`K`), list of nodes (`nodes`), and
list of elements (`elems`). In general the rows are for elements and the columns for
coordinates in x, y, etc. order.

    ^y
    |
    o--->x (longitudinal axis of the girder)

    o============o===========o ...
    ^   span_1   ^  span_2   ^

# Arguments

- `spans::AbstractArray{Float64, 1}`: Span lengths of the girder.
- `flexural_stiffnesses`

# Keywords

- `support_dofs`: Boundary conditions at the ends of each span.
    Row `i` defines the dofs of support `i` from left to right:
    `[y_translation, z_rotation]`. By default only `y_translation` is fixed (`0`).
- `num_elem_per_span`: a vector with elements as the number of elements per span.
- `additional_node_positions`: Use to place nodes
    at positions needed later, e.g. load positions and/or sensor positions.
- `hinge_positions`:  vector of internal hinge positions for the girder.

# Returns

- `Ks`: global stiffness matrix with support boundary conditions applied.
- `nodes`:
- `idx_keep`:

"""
function fem_prismatic_single_girder(
    spans::AbstractArray{Float64,1},
    flexural_stiffnesses::AbstractArray{Float64,1};
    support_dofs::AbstractArray{Int64}=zeros(Int, 0, 2),
    num_elem_per_span::AbstractArray{Int64,1}=zeros(Int, 0),
    additional_node_positions::AbstractArray{Float64,1}=zeros(0),
    hinge_positions::AbstractArray{Float64,1}=zeros(0),
)
    # ................................................
    # Initialize
    # ................................................
    if length(support_dofs) == 0
        support_dofs = repeat([1, 0]', length(spans) + 1)
    end

    if length(num_elem_per_span) == 0
        num_elem_per_span = fill(6, length(spans))
    end

    append!(additional_node_positions, hinge_positions)

    # ................................................
    # Create nodes
    # ................................................
    # Evenly distributed nodes within each span
    num_span = length(spans)
    support_positions = [0, cumsum(spans)...]
    cum_num_nodes = [0, cumsum(num_elem_per_span)...] .+ 1

    nodes = zeros(Float64, sum(num_elem_per_span) + 1, 3)
    for ii in 1:num_span
        idx_start = cum_num_nodes[ii]
        idx_end = cum_num_nodes[ii + 1]
        nodes_tmp_ii = range(support_positions[ii], stop=support_positions[ii + 1],
            length=num_elem_per_span[ii] + 1) |> collect
        nodes[idx_start:idx_end, 1] = nodes_tmp_ii
    end

    # Additional node positions, if any
    if length(additional_node_positions) != 0
        num_node_add = length(additional_node_positions)
        nodes_add = zeros(Float64, num_node_add, 3)
        nodes_add[:, 1] = additional_node_positions
        nodes = sort(unique(vcat(nodes, nodes_add), dims=1), dims=1)
    end

    num_node = size(nodes, 1)
    num_elem = num_node - 1
    num_dof = num_node * 2
    num_support = num_span + 1

    # ................................................
    # Elements
    # ................................................
    # structure of `elem_nodes`: [start_node_num, end_node_num]
    elem_nodes = _element_nodes(num_node)
    start_nodes = elem_nodes[:, 1]
    end_nodes = elem_nodes[:, 2]
    elem_EI = zeros(Float64, num_elem)

    elem_center_x = (nodes[start_nodes, 1] + nodes[end_nodes, 1]) / 2
    for ii in 1:num_span
        idx = support_positions[ii] .< elem_center_x .< support_positions[ii + 1]
        elem_EI[idx] .= flexural_stiffnesses[ii]
    end

    # ................................................
    # Internal hinges
    # ................................................
    hinges = zeros(Int, num_elem)
    if length(hinge_positions) > 0
        nodes_x = nodes[:, 1]
        hinge_nodes = [
            findfirst(nodes_x .== hinge_position) for hinge_position in hinge_positions
        ]
        hinges[hinge_nodes .- 1] .= 1
    end

    # ................................................
    # Assemble the stiffness mx
    # ................................................
    K = _assemble_K(nodes, elem_nodes, elem_EI, hinges)
    Ks, idx_keep = _apply_bcs(K, nodes, support_positions, support_dofs)

    return Ks, nodes, idx_keep
end


"""
A version of `fem_prismatic_single_girder` where the node positions (`node_xs`) and
element flexural stiffness (`elem_EI`) are provided.
"""
function fem_general_single_girder(
    node_xs::AbstractArray{Float64,1},
    elem_EI::AbstractArray{Float64,1},
    support_positions::AbstractArray{Float64,1};
    support_dofs::AbstractArray{Int64}=zeros(Int, 0, 2),
    hinge_positions::AbstractArray{Float64,1}=zeros(0),
)
    # ................................................
    # Initialize
    # ................................................
    num_node = length(node_xs)
    num_elem = length(elem_EI)
    num_dof = num_node * 2
    num_support = length(support_positions)

    support_dofs = _fem_general_input_check(
        node_xs, elem_EI, support_positions, support_dofs
    )

    if length(hinge_positions) > 0
        hinge_nodes = [findfirst(pos == node_xs) for pos in hinge_positions]
        if length(hinge_nodes) == 0 | length(hinge_nodes) != length(hinge_positions)
            throw(ArgumentError(
                "Not all `hinge_positions` correspond to an element of `node_xs`!")
            )
        end
    end

    # ................................................
    # Nodes
    # ................................................
    nodes = zeros(num_node, 3)
    nodes[:, 1] = node_xs

    # ................................................
    # Elements
    # ................................................
    # structure of `elem_nodes`: [start_node_num, end_node_num]
    elem_nodes = _element_nodes(num_node)

    # ................................................
    # Internal hinges
    # ................................................
    hinges = zeros(Int, num_elem)
    if length(hinge_positions) > 0
        nodes_x = nodes[:, 1]
        hinge_nodes = [
            findfirst(nodes_x .== hinge_position) for hinge_position in hinge_positions
        ]
        hinges[hinge_nodes .- 1] .= 1
    end

    # ................................................
    # Assemble the stiffness mx and apply bcs
    # ................................................
    K = _assemble_K(nodes, elem_nodes, elem_EI, hinges)
    Ks, idx_keep = _apply_bcs(K, nodes, support_positions, support_dofs)

    return Ks, nodes, idx_keep
end


function _fem_general_input_check(node_xs, elem_EI, support_positions, support_dofs)
    num_node = length(node_xs)
    num_elem = length(elem_EI)
    num_support = length(support_positions)

    if num_elem != num_node - 1
        throw(ArgumentError(
                "The length of `elem_EI` should be 1 less than that of `node_xs`"
            )
        )
    end

    support_in_nodes = [sp in node_xs for sp in support_positions]
    if !all(support_in_nodes)
        throw(ArgumentError(
                "The " * string((1:num_support)[.!support_in_nodes]) *
                " elements of `support_positions` do not match any `node_xs` values."
            )
        )
    end

    if length(support_dofs) > 0
        if size(support_dofs, 1) != length(support_positions)
            throw(ArgumentError(
                "`support_dofs` should have the same number of rows as the length of
                `support_positions`."
                )
            )
        end
    end

    if length(support_dofs) == 0
        support_dofs = repeat([1, 0]', length(support_positions))
    end

    return support_dofs
end


"""
    _assemble_K(nodes, elem_nodes, elem_EI, hinges) -> K

Assemble the global stiffness matrix (`K`) without regard to the boundary conditions.

# Arguments

- `nodes`: node matrix (`N×3`), each row is a node and the columns are: x, y, and z
    coordinates.
- `elem_nodes`: element connectivity matrix (`(N-1)×2`), each row is an element and the
    columns are: first node number; second node number (in correspondence with `nodes`).
- `elem_EI`: element flexural stiffness vector (`N-1`).
- `hinges`: vector indicating if an element has a hinge at one of its ends or not (`N-1`).

# Returns

- `K`: global stiffness matrix (`(2*N)×(2*N)`).

"""
function _assemble_K(nodes, elem_nodes, elem_EI, hinges)
    # ................................................
    # Initialize
    # ................................................
    num_node = size(nodes, 1)
    num_elem = size(elem_nodes, 1)
    num_dof = num_node * 2

    # ................................................
    # Assemble the structural stiffness matrix
    # ................................................
    K = zeros(Float64, num_dof, num_dof)
    for ii in 1:num_elem
        exy = nodes[elem_nodes[ii, :], 1:2]
        Ke = bernoulli_beam_2d(exy[:, 1], exy[:, 2], elem_EI[ii], hinges[ii])

        elem_dof_idx = (2 * ii - 1):(2 * ii + 2)
        K[elem_dof_idx, elem_dof_idx] += Ke
    end

    return K
end


"""
    _apply_bcs(K, nodes, support_positions, support_dofs) -> (Ks, idx_keep)

Apply boundary conditions at the supports to the stiffness matrix (`K`). Remove fully
constrained dofs (rows and columns in the stiffness matrix). The resulting matrix is `Ks`
that relates to `K` the following way: `Ks = K[idx_keep, idx_keep]`.

Note that if multiple nodes are at a support position the `support_dofs` will be applied to
all.
"""
function _apply_bcs(K, nodes, support_positions, support_dofs)
    # Get the indices (of rows and columns) to be removed (fully constrained dofs)
    num_support = length(support_positions)
    idx_remove = []
    for ii in 1:num_support
        support_dof = support_dofs[ii, :]
        if any(support_dof .!= 0)
            idxs = 2 * findall(nodes[:, 1] .≈ support_positions[ii]) .- 1
            if support_dof[1] == 1
                append!(idx_remove, idxs)
            end
            if support_dof[2] == 1
                append!(idx_remove, idxs .+ 1)
            end
        end
    end

    # Drop the constrained dofs (rows and columns)
    num_dof = size(K, 1)
    f(x) = x .!= (1:num_dof)
    if isempty(idx_remove)
        idx_keep = 1:num_dof
    else
        idx_keep = vec(prod(hcat(f.(idx_remove)...), dims=2))
    end

    Ks = K[idx_keep, idx_keep]

    return Ks, idx_keep
end


"""
    _linear_index_spring(node_pairs, num_global_dof; type)

Linear indexing of the rows and columns in the global stiffness matrix corresponding to the
nodes (`node_pairs`) connected by the springs. Translational (`type="trans"`) and rotational
(`type="rot"`) springs are available.
"""
function _linear_index_spring(node_pairs, num_global_dof; type)
    if type == "trans"
        offset = -1
    elseif type == "rot"
        offset = 0
    end

    num_spring = size(node_pairs, 1)
    idx_springs_half = 2 .* node_pairs .+ offset
    idx_springs = vcat(idx_springs_half, reverse(idx_springs_half, dims=2))
    lin_idx_off_diag = linear_index(idx_springs[:, 1], idx_springs[:, 2], num_global_dof)
    lin_idx_diag = linear_index(idx_springs[:, 1], idx_springs[:, 1], num_global_dof)

    return lin_idx_off_diag, lin_idx_diag
end


function _element_nodes(num_node)
    num_elem = num_node - 1
    start_nodes = 1:(num_node - 1)
    end_nodes = 2:num_node
    # structure of `elem_nodes`: [start_node_num, end_node_num]
    elem_nodes = zeros(Int64, num_elem, 2)
    elem_nodes[:, 1] = start_nodes
    elem_nodes[:, 2] = end_nodes
    return elem_nodes
end


function _spring_nodes(nodes, spring_positions)
    num_spring = length(spring_positions)
    spr_nodes = Array{Int64}(undef, num_spring, 2)
    for ii in 1:num_spring
        spr_nodes[ii, :] = findall(nodes[:, 1] .≈ spring_positions[ii])
    end
    return spr_nodes
end


function bending_moments_twin(u, nodes, elem_EI)
    num_node_g::Int64 = size(nodes, 1) / 2

    nodes_g1 = nodes[1:2:end, :]
    u_g1 = Array{Float64}(undef, num_node_g * 2)
    u_g1[1:2:end] = u[1:4:end]
    u_g1[2:2:end] = u[2:4:end]
    elem_EI_g1 = elem_EI[1:2:end]

    nodes_g2 = nodes[2:2:end, :]
    u_g2 = Array{Float64}(undef, num_node_g * 2)
    u_g2[1:2:end] = u[3:4:end]
    u_g2[2:2:end] = u[4:4:end]
    elem_EI_g2 = elem_EI[2:2:end]

    M = Array{Float64}(undef, num_node_g * 2)
    M[1:2:end] = bending_moments(u_g1, nodes_g1, elem_EI_g1)
    M[2:2:end] = bending_moments(u_g2, nodes_g2, elem_EI_g2)
    return M
end


"""
    bending_moments(u, nodes, elem_EI) -> M

Obtain bending moment at nodes from the complete FEA displacement field `u` (including dofs
with boundary conditions).
"""
function bending_moments(
    u,
    nodes,
    elem_EI,
    hinges::AbstractArray{Int64}=zeros(Int, 0)
)
    # ................................................
    # Initialize
    # ................................................
    num_node = size(nodes, 1)

    if length(hinges) == 0
        hinges = zeros(Int, length(elem_EI))
    end

    # structure of `elem_nodes`: [start_node_num, end_node_num]
    elem_nodes = _element_nodes(num_node)

    num_elem = size(elem_nodes, 1)
    num_dof = num_node * 2

    # ................................................
    # Assemble the bending moment vector
    # ................................................
    M = Array{Float64}(undef, num_node)
    for ii in 1:num_elem
        ue = u[(1 + 2 * (ii - 1)):(4 + 2 * (ii - 1))]
        exy = nodes[elem_nodes[ii, :], 1:2]
        me = bernoulli_beam_2d_moment(exy[:, 1], exy[:, 2], elem_EI[ii], hinges[ii])
        Me = me * ue
        if ii == num_elem
            M[ii:end] = Me
        else
            M[ii] = Me[1]
        end
    end

    return M
end
