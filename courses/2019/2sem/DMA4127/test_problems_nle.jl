const n = 1000

function f_extended_rosenbrock(x)

    f = zeros(n)

    f_extended_rosenbrock!(x, f)

    return f

end

function j_extended_rosenbrock(x)

    jac = zeros(n, n)

    j_extended_rosenbrock!(x, jac)

    return jac

end

function f_extended_rosenbrock!(x, f)

    for i = 1:2:n
        
        f[i]     = 10.0 * (x[i + 1] - x[i]^2)
        f[i + 1] = 1.0 - x[i]
        
    end

end

function j_extended_rosenbrock!(x, jac)

    for i = 1:2:n

        jac[i, i]         = - 20.0 * x[i]
        jac[i, i + 1]     =   10.0
        jac[i + 1, i]     = - 1.0
        jac[i + 1, i + 1] =   0.0

    end

end
