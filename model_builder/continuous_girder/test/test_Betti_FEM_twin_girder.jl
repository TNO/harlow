#= Testing the calculation of influence lines using moving load and Betti's theorem. For
 the plots you need to run the code snippets manually in the REPL. =#

using Test
using LinearAlgebra
using SparseArrays
using Interpolations
using Plots
using BenchmarkTools

include(abspath("continuous_girder\\FEM_girder.jl"))

@testset "three_span" begin
    # Compares two influence line computation methods:
    #  * moving a unit force along the beam, node by node
    #  * using Betti's theorem, requires a single FE analysis.

    # -------------------------------------
    # Settings & control
    # -------------------------------------
    EI = 2.0
    spans = 4.0 * ones(3)
    EIs = EI * ones(3)

    sensor_position = spans[1] / 3
    num_elem_span = 60

    spring_positions = range(0, stop=sum(spans), length=3 * length(spans)) |> collect
    spring_stiffnesses = 1e3 * ones(length(spring_positions))
    # -------------------------------------
    # Moving a unit load along the bridge
    # -------------------------------------
    Ks, nodes, idx_keep, _ = fem_prismatic_twin_girder(
        spans, EIs;
        num_elem_per_span=fill(num_elem_span, length(spans)),
        spring_positions=spring_positions,
        spring_stiffnesses=spring_stiffnesses,
        additional_node_positions=[sensor_position],
    )

    x_ml = nodes[1:2:end, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    num_node_g = convert(Int, num_node / 2)
    sensor_node = findfirst(sensor_position .== nodes[:, 1])
    elem_EI = EI * ones(num_node - 2)
    num_elem_ml = length(elem_EI)

    function moving_load()
        # pre-factorize for efficiency
        spar_Ks = sparse(Ks)
        fact_spar_Ks = factorize(spar_Ks)

        m_ml = Array{Float64}(undef, num_node_g)
        for ii = 1:num_node_g
            f = zeros(Float64, num_node * 2)
            f[1 + (ii - 1) * 4] = 1.0
            fs = f[idx_keep]

            # solve
            us = fact_spar_Ks \ fs

            # post-process
            u = zeros(Float64, num_node * 2)
            u[idx_keep] = us

            # moments
            m = bending_moments_twin(u, nodes, elem_EI)
            m_ml[ii] = m[sensor_node]
            # plot!(x_ml, u[1:4:end])
        end
        return m_ml
    end

    m_ml = moving_load()
    t_ml = @belapsed $moving_load();

    # -------------------------------------
    # Betti's theorem
    # -------------------------------------
    hinge_left_pos = sensor_position - eps(Float16)
    Ks, nodes, idx_keep, _ = fem_prismatic_twin_girder(
        spans, EIs;
        num_elem_per_span=fill(num_elem_span, length(spans)),
        spring_positions=spring_positions,
        spring_stiffnesses=spring_stiffnesses,
        additional_node_positions=[hinge_left_pos, sensor_position],
        left_hinge_positions=[sensor_position],
    )

    x_b = nodes[1:2:end, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    hinge_node = findfirst(sensor_position .== nodes[:, 1])
    hinge_left_node = findfirst(hinge_left_pos .== nodes[:, 1])
    elem_EI = EI * ones(num_node - 2)

    function betti()
        spar_Ks = sparse(Ks)
        f = zeros(Float64, num_node * 2)
        f[hinge_left_node * 2] = 1.0
        f[hinge_node * 2] = -1.0
        fs = f[idx_keep]

        # solve
        us = spar_Ks \ fs

        # post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        ϕ_hinge = u[hinge_left_node * 2] - u[hinge_node * 2]
        # scale to unit hinge rotation
        m_b = 1 / ϕ_hinge * u[1:4:end]
        return m_b
    end

    m_b = betti()
    t_b = @belapsed $betti();

    # -------------------------------------
    # Compare
    # -------------------------------------

    # Test
    itp_b = LinearInterpolation(x_b, m_b)
    m_bc = itp_b(x_ml)
    @test m_ml ≈ m_bc rtol = 1e-3

    # Print run-time
    sigdigits = 3
    println("
 ----------------------------------------------------------
    Beam ($num_elem_ml finite elements) influence line
 ----------------------------------------------------------
    ")
    println("Moving load:       ", round(t_ml * 1e3, sigdigits=sigdigits),
        " ms -> baseline")
    println("Betti's theorem:   ", round(t_b * 1e3, sigdigits=sigdigits), " ms -> ",
        round(t_ml / t_b, sigdigits=sigdigits), "× speedup")

    # Visualize
    begin
        plot(x_ml, m_ml,
            xlabel="distance", ylabel="bending moment", yflip=true,
            title="influence line", label="moving load",
            legendtitle="approach", legend=:bottomright, linewidth=2)
        plot!(x_b, m_b, ls=:dash, label="Betti's theorem", linewidth=2)
    end
end

@testset "tapered_clamped_pinned" begin
    # Compares two influence line computation methods:
    #  * moving a unit force along the beam, node by node
    #  * using Betti's theorem, requires a single FE analysis.

    # -------------------------------------
    # Settings & control
    # -------------------------------------
    b = 0.4
    h_left = 1.0
    h_right = 0.6
    E = 210
    EI_left = E * b * h_left^3 / 12
    EI_right = E * b * h_right^3 / 12
    span = 10.0
    num_elem = 40
    sensor_position = 4.0
    support_dofs = [1 1; 1 0]
    spring_positions = range(0, span, length=5) |> collect
    spring_stiffness = 10

    # Pre-process
    hinge_left_pos = sensor_position - eps(Float16)
    elem_node_xs = range(0.0, span, length=num_elem)
    node_xs_g = sort(unique(vcat(
        sensor_position, elem_node_xs, hinge_left_pos, spring_positions
    )))
    elem_xs_g = node_xs_g[1:end - 1] .+ diff(node_xs_g) / 2
    elem_EIs_g = EI_left .- elem_xs_g / span * (EI_left - EI_right)

    # -------------------------------------
    # Moving a unit load along the bridge
    # -------------------------------------
    Ks, nodes, idx_keep, _ = fem_general_twin_girder(
        node_xs_g, elem_EIs_g, [0, span];
        support_dofs=support_dofs,
        spring_positions=spring_positions,
        spring_stiffnesses=spring_stiffness * ones(length(spring_positions)),
    )

    x_ml = nodes[1:2:end, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    num_node_g = convert(Int, num_node / 2)
    sensor_node = findfirst(sensor_position .== nodes[:, 1])
    elem_EI = [elem_EIs_g elem_EIs_g]'[:]

    # pre-factorize for efficiency
    spar_Ks = sparse(Ks)
    fact_spar_Ks = factorize(spar_Ks)

    m_ml = Array{Float64}(undef, num_node)
    for ii = 1:num_node
        f = zeros(Float64, num_node * 2)
        f[1 + (ii - 1) * 2] = 1.0
        fs = f[idx_keep]

        # solve
        us = fact_spar_Ks \ fs

        # post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        # moments
        m = bending_moments_twin(u, nodes, elem_EI)
        m_ml[ii] = m[sensor_node]
        # plot!(x_ml, u[1:4:end])
    end

    # -------------------------------------
    # Betti's theorem
    # -------------------------------------
    Ks, nodes, idx_keep, _ = fem_general_twin_girder(
        node_xs_g, elem_EIs_g, [0, span];
        support_dofs=support_dofs,
        spring_positions=spring_positions,
        spring_stiffnesses=spring_stiffness * ones(length(spring_positions)),
        left_hinge_positions=[sensor_position],
    )

    x_b = nodes[1:2:end, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    hinge_node = findfirst(sensor_position .== nodes[:, 1])
    hinge_left_node = findfirst(hinge_left_pos .== nodes[:, 1])
    elem_EI = [elem_EIs_g elem_EIs_g]'[:]

    spar_Ks = sparse(Ks)
    f = zeros(Float64, num_node * 2)
    f[hinge_left_node * 2] = 1.0
    f[hinge_node * 2] = -1.0
    fs = f[idx_keep]

    # solve
    us = spar_Ks \ fs

    # post-process
    u = zeros(Float64, num_node * 2)
    u[idx_keep] = us

    ϕ_hinge = u[hinge_left_node * 2] - u[hinge_node * 2]
    # scale to unit hinge rotation
    m_b = 1 / ϕ_hinge * u[1:2:end]

    # -------------------------------------
    # Compare
    # -------------------------------------
    # Test
    @test m_ml ≈ m_b atol = 1e-3

    # Visualize
    lw = 2
    alpha = 0.5
    begin
        plot(x_ml, m_ml[1:2:end], color="red", alpha=alpha,
            xlabel="distance", ylabel="bending moment", yflip=true,
            title="influence line", label="moving load",
            legendtitle="approach", legend=:bottomright, linewidth=lw)
        plot!(x_b, m_ml[2:2:end], color="blue", alpha=alpha, linewidth=lw, label=nothing)
        plot!(x_b, m_b[1:2:end], color="red", ls=:dash,
            label="Betti's theorem", linewidth=lw)
        plot!(x_b, m_b[2:2:end], color="blue", ls=:dash, linewidth=lw, label=nothing)
    end
end
