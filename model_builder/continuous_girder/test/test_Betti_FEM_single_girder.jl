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
    num_elem_span = 120

    # -------------------------------------
    # Moving a unit load along the bridge
    # -------------------------------------
    Ks, nodes, idx_keep = fem_prismatic_single_girder(
        spans, EIs;
        num_elem_per_span=fill(num_elem_span, length(spans)),
        additional_node_positions=[sensor_position],
    )

    x_ml = nodes[:, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    sensor_node = findfirst(sensor_position .== nodes[:, 1])
    elem_EI = EI * ones(num_node - 1)
    num_elem_ml = num_node - 1

    function moving_load()
        # pre-factorize for efficiency
        spar_Ks = sparse(Ks)
        fact_spar_Ks = factorize(spar_Ks)

        m_ml = Array{Float64}(undef, num_node)
        for ii = 1:num_node
            f = zeros(Float64, num_node * 2)
            f[ii * 2 - 1] = 1.0
            fs = f[idx_keep]

            # solve
            us = fact_spar_Ks \ fs

            # post-process
            u = zeros(Float64, num_node * 2)
            u[idx_keep] = us

            # moments
            m = bending_moments(u, nodes, elem_EI)
            m_ml[ii] = m[sensor_node]
        end
        return m_ml
    end

    m_ml = moving_load()
    t_ml = @belapsed $moving_load();

    # -------------------------------------
    # Betti's theorem
    # -------------------------------------
    hinge_left_pos = sensor_position - eps(Float16)
    Ks, nodes, idx_keep = fem_prismatic_single_girder(
        spans, EIs;
        num_elem_per_span=fill(num_elem_span, length(spans)),
        additional_node_positions=[hinge_left_pos, sensor_position],
        hinge_positions=[sensor_position],
    )

    x_b = nodes[:, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    hinge_node = findfirst(sensor_position .== nodes[:, 1])
    hinge_left_node = findfirst(hinge_left_pos .== nodes[:, 1])
    elem_EI = EI * ones(num_node - 1)

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
        m_b = 1 / ϕ_hinge * u[1:2:end]
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
    @test m_ml ≈ m_bc rtol = 1e-4

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
        plot!(x_b, m_b, ls=:dash,
            label="Betti's theorem", linewidth=2)
    end
end

@testset "tapered_clamped_pinned" begin
    # Compares two influence line computation methods:
    #  * moving a unit force along the beam, node by node
    #  * using Betti's theorem, requires a single FE analysis
    #  * the results are also compared to AxisVM.

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

    # Pre-process
    hinge_left_pos = sensor_position - eps(Float16)
    elem_node_xs = range(0.0, span, length=num_elem)
    node_xs = sort(unique(vcat(sensor_position, elem_node_xs, hinge_left_pos)))
    elem_xs = node_xs[1:end - 1] .+ diff(node_xs) / 2
    elem_EIs = EI_left .- elem_xs / span * (EI_left - EI_right)

    # -------------------------------------
    # Moving a unit load along the bridge
    # -------------------------------------
    Ks, nodes, idx_keep = fem_general_single_girder(
        node_xs, elem_EIs, [0, span];
        support_dofs=support_dofs,
    )

    x_ml = nodes[:, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    sensor_node = findfirst(sensor_position .== nodes[:, 1])
    num_elem_ml = num_node - 1

    # pre-factorize for efficiency
    spar_Ks = sparse(Ks)
    fact_spar_Ks = factorize(spar_Ks)

    m_ml = Array{Float64}(undef, num_node)
    for ii = 1:num_node
        f = zeros(Float64, num_node * 2)
        f[ii * 2 - 1] = 1.0
        fs = f[idx_keep]

        # solve
        us = fact_spar_Ks \ fs

        # post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        # moments
        m = bending_moments(u, nodes, elem_EIs)
        m_ml[ii] = m[sensor_node]
    end

    # -------------------------------------
    # Betti's theorem
    # -------------------------------------
    Ks, nodes, idx_keep = fem_general_single_girder(
        node_xs, elem_EIs, [0, span];
        support_dofs=support_dofs,
        hinge_positions=[sensor_position],
    )

    x_b = nodes[:, 1]
    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)
    hinge_node = findfirst(sensor_position .== nodes[:, 1])
    hinge_left_node = findfirst(hinge_left_pos .== nodes[:, 1])

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
    begin
        plot(x_ml, m_ml,
            xlabel="distance", ylabel="bending moment", yflip=true,
            title="influence line", label="moving load",
            legendtitle="approach", legend=:bottomright, linewidth=2)
        plot!(x_b, m_b, ls=:dash,
            label="Betti's theorem", linewidth=2)
    end
end
