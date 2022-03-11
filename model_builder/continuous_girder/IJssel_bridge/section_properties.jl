using Plots

function area(coords)
    tl_zs = coords[1, :]
    tl_ys = coords[2, :]
    br_zs = coords[3, :]
    br_ys = coords[4, :]

    As = _area_rect.(tl_zs, tl_ys, br_zs, br_ys)
    return sum(As)
end

"""
centroid_y(coords)

Second moment of area (flexural inertia) about the `z` axis going through the centroid of
the cross section. All rectangles are assumed to be connected even if they are not touching
each other.
"""
function centroid_y(coords)
    tl_zs = coords[1, :]
    tl_ys = coords[2, :]
    br_zs = coords[3, :]
    br_ys = coords[4, :]

    As = _area_rect.(tl_zs, tl_ys, br_zs, br_ys)
    c_ys = _centroid_y_rect.(tl_ys, br_ys)
    return sum(c_ys .* As) / sum(As)
end

"""
    inertia_z(coords)

Second moment of area (flexural inertia) about the `z` axis going through the centroid of
the cross section. All rectangles are assumed to be connected even if they are not touching
each other.
"""
function inertia_z(coords)
    tl_zs = coords[1, :]
    tl_ys = coords[2, :]
    br_zs = coords[3, :]
    br_ys = coords[4, :]

    As = _area_rect.(tl_zs, tl_ys, br_zs, br_ys)
    I_zs = _inertia_z_rect.(tl_zs, tl_ys, br_zs, br_ys)
    c_ys = _centroid_y_rect.(tl_ys, br_ys)
    c_y = centroid_y(coords)
    return sum(I_zs .+ (c_ys .- c_y).^2 .* As)
end

function plot_sect(
    coords,
    aspect_ratio=:equal,
)
    tl_zs = coords[1, :]
    tl_ys = coords[2, :]
    br_zs = coords[3, :]
    br_ys = coords[4, :]

    p = plot(yflip=true, aspect_ratio=aspect_ratio)
    # p = plot(yflip=true)
    plot!(_rectangle.(tl_zs, tl_ys, br_zs, br_ys), opacity=0.5, label=false)
    scatter!([0], [0], label=false)
    return p
end

# ------------------------------------------------
# LOCAL UTILITY FUNCTIONS
# ------------------------------------------------

function _rectangle(tl_z, tl_y, br_z, br_y)
    return Plots.Shape(
        [tl_z, br_z, br_z, tl_z], [tl_y, tl_y, br_y, br_y]
    )
end

"""
    _area_rect(tl_z, tl_y, br_z, br_y)

Calculate the total area of a rectangle defined by its  top left coordinates
(`tl_z`, `tl_y`) and bottom right coordinates (`br_z`, `br_y`).
"""
function _area_rect(tl_z, tl_y, br_z, br_y)
    width = tl_z - br_z
    height = tl_y - br_y
    return width * height
end

function _centroid_y_rect(tl_y, br_y)
    return (tl_y + br_y) / 2
end

function _inertia_z_rect(tl_z, tl_y, br_z, br_y)
    width = tl_z - br_z
    height = tl_y - br_y
    return (width * height^3) / 12
end
