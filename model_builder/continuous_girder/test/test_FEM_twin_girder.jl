#= Testing `fem_prismatic_single_girder()` against analytical solutions and for convergence.
For the plots you need to run the code snippets manually in the REPL. =#

using Test
using LinearAlgebra
using SparseArrays
using Plots
using Interpolations

include(abspath("continuous_girder\\FEM_girder.jl"))

@testset "pinned_clamped_beam" begin
    # =====================================
    # Settings & control
    # =====================================
    # Some tests are comparing the results with results obtained from a third-party FE
    # software. If you change these input parameters those reference solutions should be
    # updated too.
    span = 4.0
    flexural_stiffness = 2.0

    load_position = span / 2
    load_intensity = 10

    support_dofs = [1 0; 1 1]

    num_elem_span = 10

    # =====================================
    # Analysis
    # =====================================

    # .....................................
    # Exact
    # .....................................
    # reference: https://www.awc.org/pdf/codes-standards/publications/design-aids/AWC-DA6-BeamFormulas-0710.pdf
    # Figure 17
    a = load_position
    b = span - load_position
    P = load_intensity
    l = span
    EI = flexural_stiffness

    function deflection(x)
        if x < a
            y = P * b^2 * x / (12 * EI * l^3) * (3 * a * l^2 - 2 * l * x^2 - a * x^2)
        else
            y = P * a / (12 * EI * l^3) * (l - x)^2 * (3 * l^2 * x - a^2 * x - 2 * a^2 * l)
        end

        return y
    end

    R1 = P * b^2 / (2 * l^3) * (a + 2 * l)
    M_max = R1 * a
    M_min = -P * a * b / (2 * l^2) * (a + l)
    itp = LinearInterpolation([0.0, a, l], [0.0, M_max, M_min])
    bending_moment(x) = itp(x)

    x_plot = range(0, stop=span, length=100) |> collect
    y_plot_exact = deflection.(x_plot)
    m_plot_exact = bending_moment.(x_plot)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FEM
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # .....................................
    # General
    # .....................................
    spans = [span]
    flexural_stiffnesses = [flexural_stiffness]

    function solve_FEM(fs, Ks, idx_keep, nodes, EI)
        num_node = size(nodes, 1)
        # Solve
        spar_Ks = sparse(Ks)
        us = spar_Ks \ fs

        # Post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        # Moments
        elem_EI = EI * ones(num_node - 2)
        m_fem = bending_moments_twin(u, nodes, elem_EI)
        m_fem_g1 = m_fem[1:2:end]
        m_fem_g2 = m_fem[2:2:end]

        # Deflections
        y_fem_g1 = u[1:4:end]
        y_fem_g2 = u[3:4:end]
        return y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2
    end

    # Build the finite element model with zero spring stiffnesses between the girders
    spring_positions = range(0, stop=span, length=num_elem_span + 1) |> collect
    spring_stiffnesses = zeros(size(spring_positions))
    Ks, nodes, idx_keep, lin_idx_springs = fem_prismatic_twin_girder(
        spans, flexural_stiffnesses;
        support_dofs=support_dofs,
        num_elem_per_span=fill(num_elem_span, length(spans)),
        additional_node_positions=[load_position],
        spring_positions=spring_positions,
        spring_stiffnesses=spring_stiffnesses,
    )

    # check if the nodes coincides with our spring positions
    @test spring_positions == nodes[1:2:end, 1]

    lin_idx_spring_diag = lin_idx_springs["diag"]
    lin_idx_spring_off_diag = lin_idx_springs["off_diag"]

    num_dof = size(Ks, 1)
    num_node = size(nodes, 1)

    x_fem = nodes[1:2:end, 1]
    y_exact = deflection.(x_fem)
    m_exact = bending_moment.(x_fem)

    # -------------------------------------
    # No connection between girders
    # -------------------------------------
    @testset "no-spring" begin
        # .....................................
        # Same load on both girders
        # .....................................
        load_node = findall(load_position .== nodes[:, 1])
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 .- 1] .= load_intensity
        fs = f[idx_keep]

        y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

        @test y_fem_g1 ≈ y_fem_g2 ≈ y_exact
        @test m_fem_g1 ≈ m_fem_g2 ≈ m_exact

        # .....................................
        # Load only one girder
        # .....................................
        load_node = findall(load_position .== nodes[:, 1])[1]
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 - 1] = load_intensity
        fs = f[idx_keep]

        y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

        @test y_fem_g1 ≈ y_exact
        @test y_fem_g2 ≈ zeros(size(y_exact))
        @test m_fem_g1 ≈ m_exact
        @test m_fem_g2 ≈ zeros(size(m_exact))
    end

    # -------------------------------------
    # "Rigid" spring between girders
    # -------------------------------------
    @testset "rigid-spring" begin
        # Add springs between the girders
        spring_stiffness = 1e6
        Ks[lin_idx_spring_diag] .+= spring_stiffness
        Ks[lin_idx_spring_off_diag] .-= spring_stiffness

        # .....................................
        # Same load on both girders
        # .....................................
        load_node = findall(load_position .== nodes[:, 1])
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 .- 1] .= load_intensity
        fs = f[idx_keep]

        y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

        @test y_fem_g1 ≈ y_fem_g2 ≈ y_exact
        @test m_fem_g1 ≈ m_fem_g2 ≈ m_exact

        # .....................................
        # Load only one girder
        # .....................................
        load_node = findall(load_position .== nodes[:, 1])[1]
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 - 1] = 2 * load_intensity
        fs = f[idx_keep]

        y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

        @test y_fem_g1 ≈ y_exact rtol = 1e-5
        @test y_fem_g2 ≈ y_exact rtol = 1e-5
        @test m_fem_g1 ≈ m_exact rtol = 1e-4
        @test m_fem_g2 ≈ m_exact rtol = 1e-4

        # Set back to zero spring stiffnesses
        Ks[lin_idx_spring_diag] .-= spring_stiffness
        Ks[lin_idx_spring_off_diag] .+= spring_stiffness
    end
    # -------------------------------------
    # "Soft" spring between girders
    # -------------------------------------
    @testset "soft-spring" begin
        # .....................................
        # Reference solution from AxisVM - load only one girder
        # .....................................
        x_axis = 0.0:0.4:4.0
        y_axis_g1 = [
            0, 1.1924, 2.2895, 3.1825, 3.7355, 3.7728,
            3.1750, 2.1967, 1.1542, 0.3324, 0
        ]
        m_axis_g1 = [
            0, 1.1640, 2.4952, 4.1647, 6.3401, 9.1573,
            4.6594, 0.7387, -2.7925, -6.1257, -9.4037
        ]

        y_axis_g2 = [
            0, 0.7742, 1.4438, 1.9175, 2.1312, 2.0606,
            1.7317, 1.2233, 0.6591, 0.1942, 0
        ]
        m_axis_g2 = [
            0, 1.3360, 2.5048, 3.3353, 3.6599, 3.3427,
            2.3406, 0.7613, -1.2075, -3.3743, -5.5963
        ]

        # For the springs between girders (vertical direction)
        spring_stiffness = 1.0

        # Loading
        load_node = findall(load_position .== nodes[:, 1])[1]
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 - 1] = 2 * load_intensity
        fs = f[idx_keep]
        # .....................................
        # Prismatic (`fem_prismatic_twin_girder`)
        # .....................................
        @testset "prismatic" begin
            # Add springs between the girders
            Ks[lin_idx_spring_diag] .+= spring_stiffness
            Ks[lin_idx_spring_off_diag] .-= spring_stiffness

            y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

            @test y_fem_g1 ≈ y_axis_g1 rtol = 1e-4
            @test y_fem_g2 ≈ y_axis_g2 rtol = 1e-4
            @test m_fem_g1 ≈ m_axis_g1 rtol = 1e-4
            @test m_fem_g2 ≈ m_axis_g2 rtol = 1e-4

            # Set back to zero spring stiffnesses
            Ks[lin_idx_spring_diag] .-= spring_stiffness
            Ks[lin_idx_spring_off_diag] .+= spring_stiffness
        end

        # .....................................
        # Element (`fem_general_twin_girder`)
        # .....................................
        @testset "general" begin
            support_positions = [0, cumsum(spans)...]
            node_xs_g = spring_positions
            elem_EI_g = EI * ones(length(spring_positions) - 1)
            spring_stiffnesses = spring_stiffness * ones(length(spring_positions))

            Ks, nodes, idx_keep, _ = fem_general_twin_girder(
                node_xs_g,
                elem_EI_g,
                support_positions;
                support_dofs=support_dofs,
                spring_positions=spring_positions,
                spring_stiffnesses=spring_stiffnesses,
            )

            y_fem_g1, y_fem_g2, m_fem_g1, m_fem_g2 = solve_FEM(fs, Ks, idx_keep, nodes, EI)

            @test y_fem_g1 ≈ y_axis_g1 rtol = 1e-4
            @test y_fem_g2 ≈ y_axis_g2 rtol = 1e-4
            @test m_fem_g1 ≈ m_axis_g1 rtol = 1e-4
            @test m_fem_g2 ≈ m_axis_g2 rtol = 1e-4

            # .....................................
            # Visualize
            # .....................................
            lw = 3
            # Deflection
            begin
                plot(x_axis, y_axis_g1, c=1, lw=lw, la=0.5,
                    xlabel="distance", ylabel="deflection", yflip=true,
                    label="Axis-girder 1",
                    legend=:bottomright)
                plot!(x_axis, y_axis_g2, label="Axis-girder 2", c=2, lw=lw, la=0.5)
                plot!(x_fem, y_fem_g1, label="FEM-girder 1", c=1, lw=lw, ls=:dash)
                plot!(x_fem, y_fem_g2, label="FEM-girder 2", c=2, lw=lw, ls=:dot)
            end
            # Bending moment
            begin
                plot(x_axis, m_axis_g1, c=1, lw=lw, la=0.5,
                    xlabel="distance", ylabel="bending moment", yflip=true,
                    label="Axis-girder 1",
                    legend=:bottomright)
                plot!(x_axis, m_axis_g2, label="Axis-girder 2", c=2, lw=lw, la=0.5)
                plot!(x_fem, m_fem_g1, label="FEM-girder 1", c=1, lw=lw, ls=:dash)
                plot!(x_fem, m_fem_g2, label="FEM-girder 2", c=2, lw=lw, ls=:dot)
            end
        end
    end
end
