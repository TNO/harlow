
"""
    bernoulli_beam_2d(
        ex::AbstractArray{Float64,1},
        ey::AbstractArray{Float64,1},
        EI::Float64
    ) -> Ke

Local stiffness matrix for a 2D Bernoulli-Navier beam element.

TODO:
* maybe `Ke` should be a symmetric type
"""
function bernoulli_beam_2d(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    hinge::Int
)
    L = norm([diff(ex), diff(ey)])

    if hinge == 0 # no hinge
        Ke = EI / L * [
            12 / L^2      6 / L     -12 / L^2     6 / L;
            6 / L         4         -6 / L        2;
            -12 / L^2     -6 / L    12 / L^2      -6 / L;
            6 / L         2         -6 / L        4;
        ]'
    elseif hinge == 1 # right hinge
        Ke = EI / L * [
            3 / L^2       3 / L     -3 / L^2      0;
            3 / L         3         -3 / L        0;
            -3 / L^2      -3 / L    3 / L^2       0;
            0             0         0             0;
        ]'
    elseif hinge == -1 # left hinge
        Ke = EI / L * [
            3 / L^2       0         3 / L         -3 / L^2;
            0             0         0             0;
            3 / L         0         3             -3 / L;
            -3 / L^2      0         -3 / L        3 / L^2;
        ]'
    else
        throw("Unknown hinge value: $hinge (It should be 0, 1, or -1.)")
    end
    return Ke
end

"""
    bernoulli_beam_2d_moment(ex, ey, EI) -> Me

Bending moment at the ends of a 2D Bernoulli-Navier beam element.
"""
function bernoulli_beam_2d_moment(
    ex::AbstractArray{Float64,1},
    ey::AbstractArray{Float64,1},
    EI::Float64,
    hinge::Int
)
    L = norm([diff(ex), diff(ey)])

    if hinge == 0 # no hinge
        Me = 2 * EI / L * [
            3 / L      -3 / L;
            2          -1;
            -3 / L     3 / L;
            1          -2;
        ]'
    elseif hinge == 1 # right hinge
        Me = 3 * EI / L * [
            1 / L      0;
            1          0;
            -1 / L     0;
            0          0;
        ]'
    elseif hinge == -1 # left hinge
        Me = 3 * EI / L * [
            0      -1 / L;
            0      0;
            0      1 / L;
            0      -1;
        ]'
    else
        throw("Unknown hinge value: $hinge (It should be 0, 1, or -1.)")
    end
    return Me
end
