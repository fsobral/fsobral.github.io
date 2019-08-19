using LinearAlgebra: norm
using Printf

function newton!(F!, J!, x, ε)

    # Dimension of the vector
    n = length(x)
    
    f = zeros(n)
    jac = zeros(n, n)
    
    # Compute F(x) and J(x) at the starting point and store it in
    # vector f and matrix jac
    F!(x, f)
    J!(x, jac)

    k = 0

    normf = norm(f, Inf)
    
    @printf("%5d %10.3e\n", k, normf)
    
    # Main loop - repeats while we are far from the solution
    while normf > ε

        # This is the most efficient way, in terms of memory. First,
        # we compute some factorization of the Jacobian matrix (in
        # this case, we use LU factorization). Then we solve the
        # system and store the solution in vector `f`, since it is
        # only necessary to calculate its norm.
        lufac = lu!(jac)
        ldiv!(lufac, f)

        # Compute new iteration. Here we use `-`, since we are using
        # vector `f` at the right hand side of equation
        x .= x - f

        # Compute F(x) and J(x) again, now at the new point, so the
        # iterations can continue
        k += 1
        F!(x, f)
        J!(x, jac)

        normf = norm(f, Inf)

        @printf("%5d %10.3e\n", k, normf)
    
    end

end

function newton_easy(F, J, x, ε)

    k = 0
    
    f = F(x)

    jac = J(x)

    normf = norm(f, Inf)
    
    @printf("%5d %10.3e\n", k, normf)
    
    while norm(f, Inf) > ε

        d = jac \ (-f)

        x = x + d

        k += 1
        
        f = F(x)

        jac = J(x)

        normf = norm(f, Inf)
    
        @printf("%5d %10.3e\n", k, normf)
    
    end

    return x    

end

function F(x)

    return [(x[1] - 3.0)^2 + (x[2] + 2.0)^2 - 16.0,
            (x[1] - 1.0)^2 + (x[2] - 1.0)^2 -  9.0]

end

function J(x)

    return [(2.0 * (x[1] - 3.0)) (2.0 * (x[2] + 2.0));
            (2.0 * (x[1] - 1.0)) (2.0 * (x[2] - 1.0))]

end

# Memory efficient implementations

function F!(x, f)

    f[1] = (x[1] - 3.0)^2 + (x[2] + 2.0)^2 - 16.0
    f[2] = (x[1] - 1.0)^2 + (x[2] - 1.0)^2 -  9.0

end

function J!(x, jac)

    jac[1, 1] = (2.0 * (x[1] - 3.0))
    jac[1, 2] = (2.0 * (x[2] + 2.0))
    jac[2, 1] = (2.0 * (x[1] - 1.0))
    jac[2, 2] = (2.0 * (x[2] - 1.0))

end
