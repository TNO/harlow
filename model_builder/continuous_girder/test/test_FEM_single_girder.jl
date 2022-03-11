#= Testing `fem_prismatic_single_girder()` against analytical solutions and for convergence.
For the plots you need to run the code snippets manually in the REPL. =#

using Test
using LinearAlgebra
using SparseArrays
using Plots
using Interpolations

include(abspath("continuous_girder\\FEM_girder.jl"))

@testset "continuous" begin
    @testset "two_equal_span_cont_beam" begin
        # -------------------------------------
        # Settings & control
        # -------------------------------------
        spans = 4.0 * ones(2)
        flexural_stiffnesses = 2.0 * ones(2)

        load_position = spans[1] / 2
        load_intensity = 10

        num_elems_span = [2, 3, 5, 8]

        # -------------------------------------
        # Analysis
        # -------------------------------------
        # reference: http://structx.com/Beam_Formulas_030.html
        y_max_exact = 0.015 * load_intensity * spans[1]^3 / flexural_stiffnesses[1]

        n = length(num_elems_span)
        p = Array{Any}(undef, n)
        y_max = Array{Float64}(undef, n)
        pp = plot()
        for ii in 1:n
            num_elem_span = num_elems_span[ii]

            Ks, nodes, idx_keep = fem_prismatic_single_girder(
                spans, flexural_stiffnesses;
                num_elem_per_span=fill(num_elem_span, length(spans)),
                additional_node_positions=[load_position]
            )

            num_dof = size(Ks, 1)
            num_node = size(nodes, 1)

            load_node = findfirst(load_position .== nodes[:, 1])
            f = zeros(Float64, num_node * 2)
            f[load_node * 2 - 1] = load_intensity
            fs = f[idx_keep]

            spar_Ks = sparse(Ks)
            fact_spar_Ks = factorize(spar_Ks)
            us = fact_spar_Ks \ fs

            # Post-process
            u = zeros(Float64, num_node * 2)
            u[idx_keep] = us

            x = nodes[:, 1]
            y = u[1:2:end, 1]
            y_max[ii] = maximum(y)

            # Test
            @test y_max_exact ≈ y_max[ii] rtol = 5e-3  # because the exact is just an approximation..

            pp = plot!(x, y,
                xlabel="distance", ylabel="deflection", yflip=true,
                label="$num_elem_span",
                legendtitle="Elem per span", legend=:bottomright,
                linewidth=2)
        end

        # Visualize
        plot(pp)
    end

    @testset "pinned_clamped_beam" begin
        # -------------------------------------
        # Settings & control
        # -------------------------------------
        span = 4.0
        flexural_stiffness = 2.0

        load_position = span / 2
        load_intensity = 10

        support_dofs = [1 0; 1 1]
        num_elem_span = 10

        # -------------------------------------
        # Analysis
        # -------------------------------------
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

        function solve_FEM(Ks, idx_keep, nodes, EI)
            num_dof = size(Ks, 1)
            num_node = size(nodes, 1)
            load_node = findfirst(load_position .== nodes[:, 1])
            elem_EI = EI * ones(num_node - 1)

            spar_Ks = sparse(Ks)
            f = zeros(Float64, num_node * 2)
            f[load_node * 2 - 1] = load_intensity
            fs = f[idx_keep]

            # solve
            us = spar_Ks \ fs

            # post-process
            u = zeros(Float64, num_node * 2)
            u[idx_keep] = us

            y_fem = u[1:2:end, 1]
            m_fem = bending_moments(u, nodes, elem_EI)
            return y_fem, m_fem
        end

        # .....................................
        # FEM
        # .....................................
        @testset "prismatic" begin
            spans = [span]
            flexural_stiffnesses = [flexural_stiffness]

            Ks, nodes, idx_keep = fem_prismatic_single_girder(
                spans, flexural_stiffnesses;
                support_dofs=support_dofs,
                num_elem_per_span=fill(num_elem_span, length(spans)),
                additional_node_positions=[load_position]
            )

            x_fem = nodes[:, 1]
            y_fem, m_fem = solve_FEM(Ks, idx_keep, nodes, EI)

            # .....................................
            # Test
            # .....................................
            y_node_exact = deflection.(x_fem)
            @test y_fem ≈ y_node_exact

            m_node_exact = bending_moment.(x_fem)
            @test m_fem ≈ m_node_exact

            # .....................................
            # Visualize
            # .....................................
            lw = 3
            # Deflection
            begin
                plot(x_plot, y_plot_exact,
                    xlabel="distance", ylabel="deflection", yflip=true,
                    label="exact",
                    legend=:bottomright,
                    linewidth=lw)
                plot!(x_fem, y_fem, label="FEM", linewidth=lw, linestyle=:dash)
            end
            # Bending moment
            begin
                plot(x_plot, m_plot_exact,
                    xlabel="distance", ylabel="bending moment", yflip=true,
                    label="exact",
                    legend=:bottomright,
                    linewidth=lw)
                plot!(x_fem, m_fem, label="FEM", linewidth=lw, linestyle=:dash)
            end
        end

        @testset "general" begin
            node_xs = range(0, span, length=20) |> collect
            node_xs = vcat(node_xs, load_position) |> unique |> sort
            elem_EI = EI * ones(length(node_xs)-1)
            support_positions=[0, span]

            Ks, nodes, idx_keep = fem_general_single_girder(
                node_xs, elem_EI, support_positions;
                support_dofs=support_dofs,
            )

            x_fem = nodes[:, 1]
            y_fem, m_fem = solve_FEM(Ks, idx_keep, nodes, EI)

            # .....................................
            # Test
            # .....................................
            y_node_exact = deflection.(x_fem)
            @test y_fem ≈ y_node_exact

            m_node_exact = bending_moment.(x_fem)
            @test m_fem ≈ m_node_exact
        end
    end
end

@testset "with_one_hinge" begin
    @testset "two_span_beam_with_a_hinge" begin
        # simply supported continuous beam with a hinge in the first span
        # -------------------------------------
        # Settings & control
        # -------------------------------------
        EI = 2.0
        spans = 4.0 * ones(2)
        EIs = EI * ones(2)

        hinge_position = spans[1] / 2 # should remain in the first span
        num_elem_span = 10

        load_position = hinge_position
        load_intensity = 10

        # -------------------------------------
        # Analysis
        # -------------------------------------
        # .....................................
        # Exact
        # .....................................
        # https://structx.com/Beam_Formulas_026.html
        a = spans[1] - hinge_position
        l = spans[2]
        y_hinge = load_intensity * a^2 / (3 * EI) * (l + a)
        function y_exact_fun(x)
            if x <= hinge_position # left side of the hinge
                y = 0.0 + y_hinge * x / hinge_position
            elseif hinge_position < x <= spans[1] # overhang
                x1 = spans[1] - x
                y = load_intensity * x1 / (6 * EI) * (2 * a * l + 3 * a * x1 - x1^2)
            else # between supports
                x1 = sum(spans) - x
                y = -load_intensity * a * x1 / (6 * EI * l) * (l^2 - x1^2)
            end
        end

        m_exact_fun = LinearInterpolation(
            [0.0, load_position, spans[1], spans[1] + spans[2]],
            [0.0, 0.0, -load_intensity * (spans[1] - load_position), 0.0])

        # .....................................
        # FEM
        # .....................................
        Ks, nodes, idx_keep = fem_prismatic_single_girder(
            spans, EIs;
            num_elem_per_span=fill(num_elem_span, length(spans)),
            additional_node_positions=[load_position],
            hinge_positions=[hinge_position],
        )

        num_dof = size(Ks, 1)
        num_node = size(nodes, 1)
        load_node = findfirst(load_position .== nodes[:, 1])
        elem_EI = EI * ones(num_node - 1)
        hinge_node = findfirst(hinge_position .== nodes[:, 1])

        hinges = zeros(Int, num_node - 1)
        # should be a right hinged element as that is what `fem_prismatic_single_girder`
        # uses
        hinges[hinge_node - 1] = 1

        spar_Ks = sparse(Ks)
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 - 1] = load_intensity
        fs = f[idx_keep]

        # solve
        us = spar_Ks \ fs

        # post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        x_fem = nodes[:, 1]
        y_fem = u[1:2:end, 1]
        m_fem = bending_moments(u, nodes, elem_EI, hinges)

        # .....................................
        # Test
        # .....................................
        y_exact = y_exact_fun.(x_fem)
        @test y_fem ≈ y_exact

        m_exact = m_exact_fun.(x_fem)
        @test m_fem ≈ m_exact

        # .....................................
        # Visualize
        # .....................................
        lw = 3
        # Deflection
        begin
            vline([hinge_position], ls=:dashdot, label=nothing)
            plot!(x_fem, y_exact,
                xlabel="distance", ylabel="deflection", yflip=true,
                label="exact",
                legend=:bottomright,
                linewidth=lw)
            plot!(x_fem, y_fem, label="FEM", linewidth=lw, linestyle=:dash)
        end

        # Bending moment
        begin
            vline([hinge_position], ls=:dashdot, label=nothing)
            plot!(x_fem, m_exact,
                xlabel="distance", ylabel="bending moment", yflip=true,
                label="exact",
                legend=:bottomright,
                linewidth=lw)
            plot!(x_fem, m_fem, label="FEM", linewidth=lw, linestyle=:dash)
        end
    end

    @testset "one_span_beam_with_a_hinge" begin
        # clamped-hinged end beam with a hinge in the span
        # -------------------------------------
        # Settings & control
        # -------------------------------------
        EI = 2.0
        span = 4.0

        hinge_position = span * 3 / 4
        num_elem = 10

        load_position = hinge_position * 3 / 4 # stay on the left side of the hinge
        load_intensity = 30

        support_dofs = [1 1; 1 0]
        # -------------------------------------
        # Analysis
        # -------------------------------------
        # .....................................
        # Exact
        # .....................................
        # https://structx.com/Beam_Formulas_021.html
        a = hinge_position - load_position
        b = load_position
        l = a + b
        y_hinge = load_intensity * b^2 / (6 * EI) * (3 * l - b)
        function y_exact_fun(x)
            if x <= load_position
                x1 = l - x
                y = load_intensity * (l - x1)^2 / (6 * EI) * (3 * b - l + x1)
            elseif load_position < x <= hinge_position
                x1 = l - x
                y = load_intensity * b^2 / (6 * EI) * (3 * l - 3 * x1 - b)
            else # right side of the support
                x1 = x - hinge_position
                y = y_hinge * (1 - x1 / (span - hinge_position))
            end
        end

        m_exact_fun = LinearInterpolation(
            [0.0, load_position, span],
            [-load_position * load_intensity, 0.0, 0.0])

        # .....................................
        # FEM
        # .....................................
        Ks, nodes, idx_keep = fem_prismatic_single_girder(
            [span], [EI];
            num_elem_per_span=[num_elem],
            additional_node_positions=[load_position],
            hinge_positions=[hinge_position],
            support_dofs=support_dofs,
        )

        num_dof = size(Ks, 1)
        num_node = size(nodes, 1)
        load_node = findfirst(load_position .== nodes[:, 1])
        elem_EI = EI * ones(num_node - 1)
        hinge_node = findfirst(hinge_position .== nodes[:, 1])

        hinges = zeros(Int, num_node - 1)
        # should be a right hinged element as that is what `fem_prismatic_single_girder`
        # uses
        hinges[hinge_node - 1] = 1

        spar_Ks = sparse(Ks)
        f = zeros(Float64, num_node * 2)
        f[load_node * 2 - 1] = load_intensity
        fs = f[idx_keep]

        # solve
        us = spar_Ks \ fs

        # post-process
        u = zeros(Float64, num_node * 2)
        u[idx_keep] = us

        x_fem = nodes[:, 1]
        y_fem = u[1:2:end, 1]
        m_fem = bending_moments(u, nodes, elem_EI, hinges)

        # General function to assemble the stiffness matrix
        Ks_gen, nodes_gen, idx_keep_gen = fem_general_single_girder(
            nodes[:, 1], EI*ones(size(nodes, 1) - 1), [0, span];
            support_dofs=support_dofs,
            hinge_positions=[hinge_position],
        )

        # .....................................
        # Test
        # .....................................
        @test Ks ≈ Ks_gen
        @test nodes ≈ nodes_gen
        @test idx_keep == idx_keep_gen

        y_exact = y_exact_fun.(x_fem)
        @test y_fem ≈ y_exact

        m_exact = m_exact_fun.(x_fem)
        @test m_fem ≈ m_exact

        # .....................................
        # Visualize
        # .....................................
        lw = 3
        # Deflection
        begin
            vline([hinge_position], ls=:dashdot, label=nothing)
            plot!(x_fem, y_exact,
                xlabel="distance", ylabel="deflection", yflip=true,
                label="exact",
                legend=:bottomright,
                linewidth=lw)
            plot!(x_fem, y_fem, label="FEM", linewidth=lw, linestyle=:dash)
        end

        # Bending moment
        begin
            vline([hinge_position], ls=:dashdot, label=nothing)
            plot!(x_fem, m_exact,
                xlabel="distance", ylabel="bending moment", yflip=true,
                label="exact",
                legend=:bottomright,
                linewidth=lw)
            plot!(x_fem, m_fem, label="FEM", linewidth=lw, linestyle=:dash)
        end
    end
end
