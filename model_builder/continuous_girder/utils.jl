"""
    linterp_even(xg, yg, extrap_fill=0.0)

1D linear interpolation using evenly spaced `xg` and corresponding `yg`.

For values outside of `xg` and `extrap_fill` is returned. This function returns a function
that is to be evaluated. This function is compatible with `Zygote.jl` in contrast with
`Interpolations.jl`

# Examples
```jldoctest
julia> itp = linterp_even([1, 2, 3], [1, 3, 6])
julia> itp(2.5)
4.5
```
"""
function linterp_even(xg, yg, extrap_fill=0.0)
    num_xg = length(xg)
    xg_min = minimum(xg)
    xg_max = maximum(xg)
    uxg = 1:num_xg

    function _linterp(x)
        ux = (x - xg_min) / (xg_max - xg_min) * (num_xg - 1) + 1
        idx0 = convert(Int64, floor(ux))
        idx1 = convert(Int64, ceil(ux))
        if idx0 < 1 || idx1 > num_xg
            y = extrap_fill
        else
            if idx0 == idx1
                y = yg[idx0]
            else
                x0 = xg[idx0]
                x1 = xg[idx1]
                y0 = yg[idx0]
                y1 = yg[idx1]
                y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            end
        end
        return y
    end

    return _linterp
end


"""
    relative(p::Plots.Subplot, rx, ry)

Place a `Plots.jl` annotation with relative coordinates Of the plotting area).
Source: https://github.com/JuliaPlots/Plots.jl/issues/2728#issuecomment-632916193

# Examples
```
p = plot(plot(rand(100)), plot(rand(200)), legend = false, layout = grid(1,2), frame=:box);
annotate!(sp=1, relative(p[1], 0.03, 0.95)..., text("Cats&Dogs", :left))
annotate!(sp=2, relative(p[2], 0.03, 0.95)..., text("Cats&Dogs", :left))
```
"""
function relative(p::Plots.Subplot, rx, ry)
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    return xlims[1] + rx * (xlims[2] - xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
end


"""
Convert linear indices to linear indices(subscripts) for a 2D array. See the `ind2sub`
Matlab function.
"""
function linear_index(idx_row, idx_col, num_row)
    return idx_row .+ (idx_col .- 1) * num_row
end
