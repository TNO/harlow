#=
Description:
-----------
This file defines the dimensions of a main girder of the steel IJssel bridge:
    * cross-section
    * changes of the cross-section over the bridge

Since the web height is changing it also provides a discretized representation that can be
used to build a finite element model and do structural analysis.

We build only half of the girder and at the end mirror it about its axis of symmetry.


Units:
------
    * All length dimensions are in [mm] for convenience
    * all other units are SI base units for convenience

References:
-----------
 [1] RHDHV (2020) Uitgangspuntenrapport herberekening IJsselbrug A12.
    T&P-BF7387-R001-F1.0. 1.0/Definitief

TODO:
 * for added nodes the cross section could be the accurate one?!
 * probably better for the next time: follow my typical Ansys practice: vectors for changes
    combine them and change components when looping over. =#

using DataFrames
using Test

include(abspath(".\\continuous_girder\\IJssel_bridge\\section_properties.jl"))

function girder_sections(;
    max_elem_length=100_000.0,
    additional_node_positions=[0.0],
    consider_K_braces=false,
)
    # =============================================
    # HARD CODED VALUES
    # =============================================
    support_xs = cumsum([0.0, 44991.0, 50027.0, 104964.0, 50027.0, 44991.0])

    # K-bracings (each term is a cross-beam distance, not perfectly matching [1] as it has
    # internal inconsistencies)
    K_brace_span1_dxs = [
        1800 + 1800 + 1800,
        1800 + 1800 + 1800,
        1800 + 1800 + 1800,
        1800 + 1796,
        1800 + 1800,
        1800 + 1800 + 1800,
        1800 + 1800 + 1800,
        1800 + 1795 + 1800,
        1800 + 1800 + 1800,
    ]
    K_brace_span2_dxs = [
        1800 + 1800 + 1800,
        1785 + 1785 + 1785,
        1785 + 1785 + 1785,
        1785 + 1785 + 1785,
        1785 + 1786,
        1785 + 1786,
        1785 + 1785 + 1785,
        1785 + 1785 + 1785,
        1785 + 1785 + 1785,
        1785 + 1785 + 1785,
    ]
    K_brace_span3half_dxs = [
        1750 + 1750 + 1750,
        1750 + 1750 + 1744,
        1750 + 1750 + 1750,
        1750 + 1750 + 1750,
        1750 + 1750 + 1744,
        1750 + 1750 + 1750,
        1750 + 1750 + 1750,
        1750 + 1750 + 1744,
        1750 + 1750 + 1750,
        1750 + 1750 + 1750,
    ]

    @test sum(K_brace_span1_dxs) == support_xs[2]
    @test sum(K_brace_span2_dxs) == support_xs[3] - support_xs[2]
    @test sum(K_brace_span3half_dxs) == (support_xs[4] - support_xs[3]) / 2

    K_brace_half_dxs = vcat(K_brace_span1_dxs, K_brace_span2_dxs, K_brace_span3half_dxs)
    K_brace_xs = cumsum([0.0, vcat(K_brace_half_dxs, reverse(K_brace_half_dxs))...])

    if consider_K_braces
        additional_node_positions = vcat(additional_node_positions, K_brace_xs) |>
            sort |> unique
    end

    # ---------------------------------------------
    # CROSS-SECTIONS, Z-Y (transverse dir)
    # ---------------------------------------------
    # the origin is at the intersection of the top plane of the deck and
    # the center plane of the web
    t_web = 12.0

    # .............................................
    # Stiffeners (longitudinal deck stiffeners)
    # .............................................
    # normal stiffeners, fig.92
    t_normal_stiff = 10.7 # equivalent thickness, e.g. [1] p.302
    h_normal_stiff = 160.0

    # small stiffeners (only the last stiffener), fig.92
    t_small_stiff = 8.0
    h_small_stiff = 100.0

    # .............................................
    # Stiffened deck
    # .............................................
    t_deck_normal = 10.0
    t_deck_thick = 12.0
    w_deck_end_center = 5700.0 / 2
    w_deck_end_side = 1775.0

    # centroid of stiffeners towards the side (-), [1] fig.89, seemingly in conflict with
    # the top figure of p.541
    tmp = cumsum(300 * ones(4))
    c_z_side_stiff = -[tmp..., tmp[end] + 290.0]

    # centroid of stiffeners towards the center (+), [1] fig.89
    c_z_center_stiff = cumsum(300.0 * ones(9))

    # Note: the 445x10 endplate (fig.92) is not included although maybe it should be, see
    # for example fig.107
    # .............................................
    # Top flange - half DIN 30 profile, [1] p.520
    # .............................................
    w_tf_base = 300.0
    t_tf_base = 20.0

    # .............................................
    # Bottom flange - half DIN 30 profile + additional plates
    # .............................................
    # For convenience following the color coding of RHDHV, [1] section 8
    w_bf_base = w_tf_base
    t_bf_base = t_tf_base

    # [1] section 7.4.1.2
    bottom_flange = Dict(
        "red" => Dict{String,Array}("t" => [t_bf_base, 20.0], "w" => [w_bf_base, 500.0]),
        "green" => Dict{String,Array}("t" => [t_bf_base, 30.0], "w" => [w_bf_base, 500.0]),
        "yellow" => Dict{String,Array}("t" => [t_bf_base, 30.0, 30.0], "w" => [w_bf_base, 500.0, 350.0]),
        "pink" => Dict{String,Array}("t" => [t_bf_base, 30.0, 10.0], "w" => [w_bf_base, 500.0, 530.0]),
        "blue" => Dict{String,Array}("t" => [t_bf_base, 30.0, 30.0], "w" => [w_bf_base, 500.0, 550.0]),
    )

    # =============================================
    # CROSS-SECTIONS, X (longitudinal dir)
    # =============================================
    #=
    Due to symmetry only half of the supports are provided.

    How the cross-section changes along the longitudinal axis of the bridge
    a cross-section is made up of three components:
        * top_flange (deck + DIN 30 flange)
        * web
        * bottom_flange

    If any of these change then we have a new DataFrame row (expect for linear change in
    web height).
    [1] section 8.4, 8.5; deck thickness: section 12.2, 12.3=#
    # Beginning of span 1 (pillar F)
    girder_half = DataFrame(
        "start_x" => -322.0,
        "end_x" => -322.0 + 10225.0,
        "bottom_flange" => "green",
        "h_web_start" => 2360.0,
        "h_web_end" => 2360.0,
        "t_web" => t_web,
        "top_flange" => "normal",
    )
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 23598.0,
        "bottom_flange" => "yellow",
        "h_web_start" => 2360.0,
        "h_web_end" => 2360.0,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 1500,
        "bottom_flange" => "green",
        "h_web_start" => 2360,
        "h_web_end" => 2360,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 31022,
        "bottom_flange" => "red",
        "h_web_start" => 2360,
        "h_web_end" => 2909,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 2942,
        "bottom_flange" => "green",
        "h_web_start" => 2909,
        "h_web_end" => 3091,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    # vv ....................................... vv
    start_x = girder_half.end_x[end]
    l_deck = 26053
    l_thick_deck = 6.75 * 1750
    h_web_start = 3091
    h_web_end = 5280
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + (l_deck - l_thick_deck),
        "bottom_flange" => "blue",
        "h_web_start" => h_web_start,
        "h_web_end" => h_web_start + (h_web_end - h_web_start) * (l_deck - l_thick_deck) / l_deck,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    append!(girder_half, DataFrame(
        "start_x" => girder_half.end_x[end],
        "end_x" => start_x + l_deck,
        "bottom_flange" => "blue",
        "h_web_start" => girder_half.h_web_end[end],
        "h_web_end" => h_web_end,
        "t_web" => t_web,
        "top_flange" => "thick",
    ))
    # ^^ ....................................... ^^
    # Beginning of span 3 (pillar H)
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 5909.5,
        "bottom_flange" => "blue",
        "h_web_start" => 5280,
        "h_web_end" => 4802,
        "t_web" => t_web,
        "top_flange" => "thick",
    ))
    # vv ....................................... vv
    start_x = girder_half.end_x[end]
    l_deck = 10111
    l_thick_deck = 4.5 * 1750 - 5909.5
    h_web_start = 4802
    h_web_end = 4002
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + l_thick_deck,
        "bottom_flange" => "green",
        "h_web_start" => h_web_start,
        "h_web_end" => h_web_start + (h_web_end - h_web_start) * l_thick_deck / l_deck,
        "t_web" => t_web,
        "top_flange" => "thick",
    ))
    append!(girder_half, DataFrame(
        "start_x" => girder_half.end_x[end],
        "end_x" => start_x + l_deck,
        "bottom_flange" => "green",
        "h_web_start" => girder_half.h_web_end[end],
        "h_web_end" => h_web_end,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    # ^^ ....................................... ^^
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 12293,
        "bottom_flange" => "red",
        "h_web_start" => 4002,
        "h_web_end" => 3315,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 1227,
        "bottom_flange" => "green",
        "h_web_start" => 3315,
        "h_web_end" => 3268,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    start_x = girder_half.end_x[end]
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + 5700,
        "bottom_flange" => "pink",
        "h_web_start" => 3268,
        "h_web_end" => 3061,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    # vv ....................................... vv
    start_x = girder_half.end_x[end]
    l_deck = 34483 / 2
    l_thick_deck = 9 * 1750
    h_web_start = 3061
    h_web_end = 2760
    append!(girder_half, DataFrame(
        "start_x" => start_x,
        "end_x" => start_x + (l_deck - l_thick_deck),
        "bottom_flange" => "blue",
        "h_web_start" => h_web_start,
        "h_web_end" => h_web_start + (h_web_end - h_web_start) * (l_deck - l_thick_deck) / l_deck,
        "t_web" => t_web,
        "top_flange" => "normal",
    ))
    append!(girder_half, DataFrame(
        "start_x" => girder_half.end_x[end],
        "end_x" => start_x + l_deck,
        "bottom_flange" => "blue",
        "h_web_start" => girder_half.h_web_end[end],
        "h_web_end" => h_web_end,
        "t_web" => t_web,
        "top_flange" => "thick",
    ))
    # ^^ ....................................... ^^
    # center of the middle span (axis of symmetry)

    # check if we reach the symmetry center
    @test girder_half.end_x[end] ≈ support_xs[end] / 2

    # =============================================
    # DISCRETIZE & COMPUTE SECT PROPS
    # =============================================

    # .............................................
    # Stiffened deck + top flange
    # .............................................
    n_center_stiff = length(c_z_center_stiff)
    n_side_stiff = length(c_z_side_stiff)

    function top_section(t_deck)
        # deck plate + DIN 30 flange
        tl_zs_top = [-w_deck_end_side, -w_tf_base / 2]
        tl_ys_top = [0.0, t_deck]
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
        return hcat(tl_zs_top, tl_ys_top, br_zs_top, br_ys_top)'
    end

    coords_top_normal = top_section(t_deck_normal)
    coords_top_thick = top_section(t_deck_thick)

    top_flange = Dict(
        "normal" => Dict("coords" => coords_top_normal),
        "thick" => Dict("coords" => coords_top_thick),
    )
    # plot(coords_top_normal...)
    # plot(coords_top_thick...)

    # .............................................
    # Bottom flange, to be shifted along y
    # .............................................
    # origin: top-center of the first rectangle
    tl_zs_top = Array{Float64}[]
    for (key, value) in bottom_flange
        w = value["w"]
        t = value["t"]
        n_rect = length(w)

        tl_zs = -w ./ 2
        tl_ys = [0, cumsum(t)...][1:(end - 1)]
        br_zs = w ./ 2
        br_ys = cumsum(t)

        push!(bottom_flange[key],
            "coords" => hcat(tl_zs,  tl_ys, br_zs, br_ys)',
        )
    end

    # .............................................
    # Generate the points/nodes along the girder
    # .............................................
    symmetry_x = support_xs[end] / 2

    # additional points
    # make it symmetric for convenience (at the cost of a few additional points)
    function mirror(x)
        return vcat(x, symmetry_x .+ reverse(symmetry_x .- x)) |> unique
    end

    add_xs = mirror(additional_node_positions)
    add_half_xs = add_xs[add_xs .<= symmetry_x]

    # cross-section changes
    cs_half_xs = vcat(girder_half.start_x, girder_half.end_x) |> unique

    # supports
    support_half_xs = support_xs[support_xs .<= symmetry_x]

    node_base_xs = vcat(support_half_xs, cs_half_xs, add_half_xs) |> sort |> unique

    # add nodes to meet the `max_elem_length` requirement
    node_xs = deepcopy(node_base_xs)
    l_dist = diff(node_base_xs)
    idxs = findall(l_dist .> max_elem_length)

    for idx in idxs
        interval_start = node_xs[idx]
        interval_end = node_xs[idx + 1]
        L = interval_end - interval_start
        n_elem = floor(L / max_elem_length)
        new_xs = interval_start .+ (1:n_elem) .* max_elem_length
        append!(node_xs, new_xs)
    end

    unique!(sort!(node_xs))
    n_node = length(node_xs)

    # .............................................
    # Walk along the girder and compute sect props
    # .............................................
    n_elem = n_node - 1
    elem_xs = node_xs[1:(end - 1)] .+ diff(node_xs) / 2
    elem_cs_coords = Array{Array}(undef, n_elem)
    elem_I_zs = Array{Float64}(undef, n_elem)
    elem_c_ys = Array{Float64}(undef, n_elem)
    elem_h_ys = Array{Float64}(undef, n_elem)

    for segment in eachrow(girder_half)
        start_x = segment.start_x
        end_x = segment.end_x
        t_web = segment.t_web
        h_web_start = segment.h_web_start
        h_web_end = segment.h_web_end
        top_flange_cat = segment.top_flange
        bottom_flange_cat = segment.bottom_flange
        top_flange_coords = top_flange[top_flange_cat]["coords"]
        bottom_flange_coords = bottom_flange[bottom_flange_cat]["coords"]

        idx = abs.(top_flange_coords[3, :]) .< 1.1 * w_tf_base / 2
        top_flange_bottom = maximum(top_flange_coords[4, idx])

        elem_idxs = findall(start_x .<= elem_xs .<= end_x)

        for elem_idx in elem_idxs
            elem_x = elem_xs[elem_idx]
            h_web = h_web_start + (h_web_end - h_web_start) *
                (elem_x - start_x) /  (end_x - start_x)
            web_coords = [-t_web / 2; top_flange_bottom; t_web / 2;
                top_flange_bottom + h_web]
            bf_coords = deepcopy(bottom_flange_coords)
            bf_coords[[2,4],:] .+= top_flange_bottom + h_web

            # combine section components
            sect_coords = hcat(top_flange_coords, web_coords, bf_coords)

            # collect
            elem_cs_coords[elem_idx] = sect_coords
            elem_I_zs[elem_idx] = inertia_z(sect_coords)
            elem_c_ys[elem_idx] = centroid_y(sect_coords)
            elem_h_ys[elem_idx] = maximum(sect_coords[4, :])
        end
    end

    # =============================================
    # MIRROR
    # =============================================
    node_full_xs = mirror(node_xs)
    elem_full_xs = mirror(elem_xs)
    elem_full_cs_coords = vcat(elem_cs_coords, reverse(elem_cs_coords))
    elem_full_I_zs = vcat(elem_I_zs, reverse(elem_I_zs))
    elem_full_c_ys = vcat(elem_c_ys, reverse(elem_c_ys))
    elem_full_h_ys = vcat(elem_h_ys, reverse(elem_h_ys))

    # Collect
    girder = Dict(
        "support_xs" => support_xs,
        "K_brace_xs" => K_brace_xs,
        "node_xs" => node_full_xs,
        "elem_xs" => elem_full_xs,
        "elem_cs_coords" => elem_full_cs_coords,
        "elem_I_zs" => elem_full_I_zs,
        "elem_c_ys" => elem_full_c_ys,
        "elem_h_ys" => elem_full_h_ys,
        "girder_half" => girder_half,
    )
    return girder
end
