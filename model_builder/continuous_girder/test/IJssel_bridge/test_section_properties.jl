using Test

include(abspath("continuous_girder\\IJssel_bridge\\section_properties.jl"))

@testset "simple_sections" begin
    @testset "rectangle" begin
        t = 2.0
        w = 10.0

        tl_zs = [-w / 2]
        tl_ys = [0.0]
        br_zs = [w / 2]
        br_ys = [t]
        coords = hcat(tl_zs, tl_ys, br_zs, br_ys)'

        c_y = centroid_y(coords)
        I_z = inertia_z(coords)

        @test c_y ≈ t / 2
        @test I_z ≈ w * t^3 / 12
    end
    @testset "T-section" begin
        # the reference solutions are from Axis VM
        t1 = 2.0
        w1 = 10.0

        t2 = 1.0
        w2 = 5.0

        tl_zs = [-w1 / 2, -t2 / 2]
        tl_ys = [0.0, t1]
        br_zs = [w1 / 2, t2 / 2]
        br_ys = [t1, t1 + w2]
        coords = hcat(tl_zs, tl_ys, br_zs, br_ys)'

        c_y = centroid_y(coords)
        I_z = inertia_z(coords)

        @test c_y ≈ 1.70
        @test I_z ≈ 66.083330 rtol = 1e-4
    end
    @testset "IJssel_bridge_green_2400" begin
        # the reference solutions are from Axis VM
        # .............................................
        # Basic dimensions
        # .............................................
        # web thickness and height
        t_web = 12.0
        h_web = 2400

        # normal stiffeners
        t_normal_stiff = 10.7
        h_normal_stiff = 160.0
        # small stiffeners (only the last stiffener)
        t_small_stiff = 8.0
        h_small_stiff = 100.0

        # stiffened deck plate
        t_deck = 10.0
        w_deck_end_center = 5700.0 / 2
        w_deck_end_side = 1775.0

        # centroid of stiffeners towards the side (-)
        tmp = cumsum(300 * ones(4))
        c_z_side_stiff = -[tmp..., tmp[end] + 290.0]
        # centroid of stiffeners towards the center (+)
        c_z_center_stiff = cumsum(300.0 * ones(9))

        # Top and bottom flange of the half DIN 30 profile
        w_tf_base = 300.0
        t_tf_base = 16.0
        w_bf_base = w_tf_base
        t_bf_base = t_tf_base

        # bottom additional flange
        w_bf_add = 500
        t_bf_add = 30

        # .............................................
        # Assemble the web
        # .............................................
        web_coords = [-t_web / 2; t_deck + t_tf_base; t_web / 2; t_deck + t_tf_base + h_web]

        # .............................................
        # Assemble the top part
        # .............................................
        n_center_stiff = length(c_z_center_stiff)
        n_side_stiff = length(c_z_side_stiff)
        # deck
        tl_zs_top = [-w_deck_end_side, -w_tf_base / 2]
        tl_ys_top = [0.0, t_deck]

        # DIN 30 top flange
        br_zs_top = [w_deck_end_center, w_tf_base / 2]
        br_ys_top = [t_deck, t_deck + t_tf_base]

        # center stiffeners
        append!(tl_zs_top, c_z_center_stiff .- t_normal_stiff / 2)
        append!(tl_ys_top, t_deck * ones(n_center_stiff))
        append!(br_zs_top, c_z_center_stiff .+ t_normal_stiff / 2)
        append!(br_ys_top, t_deck .+ h_normal_stiff * ones(n_center_stiff))

        # side stiffeners
        append!(tl_zs_top, c_z_side_stiff[1:(end - 1)] .- t_normal_stiff / 2)
        append!(tl_ys_top, t_deck * ones(n_side_stiff - 1))
        append!(br_zs_top, c_z_side_stiff[1:(end - 1)] .+ t_normal_stiff / 2)
        append!(br_ys_top, t_deck .+ h_normal_stiff * ones(n_side_stiff - 1))

        append!(tl_zs_top, c_z_side_stiff[end] - t_small_stiff / 2)
        append!(tl_ys_top, t_deck)
        append!(br_zs_top, c_z_side_stiff[end] .+ t_small_stiff / 2)
        append!(br_ys_top, t_deck .+ h_small_stiff)

        top_coords = hcat(tl_zs_top, tl_ys_top, br_zs_top, br_ys_top)'

        # .............................................
        # Assemble the bottom part
        # .............................................
        shift = (t_deck + t_tf_base + h_web)

        w = [w_tf_base, w_bf_add]
        t = [t_tf_base, t_bf_add]

        tl_zs_bot = -w ./ 2
        tl_ys_bot = [0, cumsum(t)...][1:(end - 1)] .+ shift
        br_zs_bot = w ./ 2
        br_ys_bot = cumsum(t) .+ shift

        bottom_coords = hcat(tl_zs_bot, tl_ys_bot, br_zs_bot, br_ys_bot)'

        # .............................................
        # Combine section components
        # .............................................
        sect_coords = hcat(top_coords, web_coords, bottom_coords)
        plot_sect(sect_coords)

        # .............................................
        # Compute section properties
        # .............................................
        I_z = inertia_z(sect_coords)
        c_y = centroid_y(sect_coords)

        @test c_y ≈ 702.6187 rtol = 1e-5
        @test I_z ≈ 1.157612 * 1e11 rtol = 1e-5
    end
end
