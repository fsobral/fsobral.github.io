# This program was developed during the DMA4127 class by
# Lucas Francisco dos Santos
#
# Maringá, November, 2019.

using LinearAlgebra
using Printf
#using Plots; pyplot()
using PyPlot

function TR_box!(F, G!, H!, x, lo, up; flag = 0, ϵ=1e-6, Δ=2., η=0.25, η2=0.75, ITMAX=1000,ttt="ro")
    #initial atributions
    k = 0
    l = length(x)
    Δk= Δ*ones(l,2) #box size [xi_l, xi_u], i = 1,...,l

    p = zeros(l)    #TR step
    ρ = 0           #model razao
    g = zeros(l)    #gradient of f
    B = zeros(l,l)  #hessian of f
    λ = zeros(2*l)  #lagrange multipliers
    gp= zeros(l)    #gradient of f projeted to trust region intersection with box constraint
    normp = 999
    
    plot_3d(Fbranin, l=(lo .- 1.0), u=(up .+ 1.0))

    if ( minimum(x-lo)<0 ||  minimum(up-x)<0 )  #infeasible initial guess
        x .= min.(up, max.(lo, x) )
    end

    PyPlot.plot([lo[1], lo[1], up[1], up[1], lo[1]],
                [lo[2], up[2], up[2], lo[2], lo[2]], "-k")
    
    @printf("ite,   F(x),    ||p||∞\n")
    while (k < ITMAX)

        PyPlot.plot([x[1]], [x[2]], ttt, markersize=5)
        PyPlot.text([x[1]], [x[2]], "$(k)")

        #initial calculation of gradient and hessians in xk
        G!(x,g)
        H!(x,B)

        #correct Δk w.r.t. upper and lower bound of x -> intersection b/w TR and box
        for i = 1:l
            if ( (x[i] - Δk[i,1]) < lo[i] )
                Δk[i,1] = x[i] - lo[i]
            end
            if ( (x[i] + Δk[i,2]) > up[i] )
                Δk[i,2] = up[i] - x[i]
            end
        end

        @printf("%02d, %4.3e, %4.3e \n",k,F(x)[1],normp)

        # 1) Calculate pk that is the approx. solution of min. mk(p), s.t. norm(p)<= Δk 
        if (flag == 0) #projection com model solution in projected direction
            mult = 0
            gp .= min.(x+Δk[:,2], max.(x-Δk[:,1], x-g) )-x
            for i = 1:l
                if (gp[i]<0)
                    mult = max(mult, abs(gp[i])/(Δk[i,1]) )
                elseif (gp[i]>0)
                    mult = max(mult, abs(gp[i])/(Δk[i,2]) )
                end
            end
            mult = 1/(mult+eps())
            p .= mult*(gp)
            #for i = 1:l
            #    if (gp[i]==x[i])
            #        p[i] = 0
            #    end
            #end
            curv = p'*B*p
            if (curv <= 0)
                τ = 1
            else
                τ = min(1, dot(-g,p)/curv )
            end

            p .= τ*p

        elseif (flag == 1) #cauchy point with projection
            curv = (-g)'*B*(-g)
            mult = 0
            if (minimum(Δk)>0)
                for i = 1:l
                    if (-g[i]<0)
                        mult = max(mult, abs.(g[i])./(Δk[i,1]) )
                    else
                        mult = max(mult, abs.(g[i])./(Δk[i,2]) )
                    end
                end
                if (curv <= 0)
                    τ = 1
                else
                    dotg = dot(g,g)
                    τ = min(1, dotg*mult/curv )
                end

                p .= -τ*g/mult
            else
            #projection in case x[i]==(up[i] or lo[i])
                #with explicit model solution in projected direction
                for i = 1:l
                    if (-g[i]<0)
                        if (g[i]>Δk[i,1])
                            p[i] = -min(Δk[i,1],g[i])
                        else
                            p[i] = -max(Δk[i,1],g[i])
                        end
                    else
                        if (-g[i]>Δk[i,2])
                            p[i] = min(Δk[i,2],-g[i])
                        else
                            p[i] = max(Δk[i,2],-g[i])
                        end
                    end
                end
                curv_proj = p'*B*p
                if (curv_proj > 0)
                    τ = min(1, dot(-g,p)/(p'*B*p) )
                else
                    τ = 1
                end
                #println(τ)
                p = τ*p

                #with line search
                #while (F(x + p)[1] - F(x)[1] >= 0 && τ>ϵ)
                #    τ = τ/2
                #    p = τ*p
                #end
            end

        elseif (flag==2) # projection only (projeção para as "quinas")         
            for i = 1:l
                if (-g[i]<0)
                    if (g[i]>Δk[i,1])
                        p[i] = -min(Δk[i,1],g[i])
                    else
                        p[i] = -max(Δk[i,1],g[i])
                    end
                else
                    if (-g[i]>Δk[i,2])
                        p[i] = min(Δk[i,2],-g[i])
                    else
                        p[i] = max(Δk[i,2],-g[i])
                    end
                end
            end
            curv_proj = p'*B*p
            if (curv_proj > 0)
                τ = min(1, dot(-g,p)/curv_proj )
            else
                τ = 1
            end
            p = τ*p
        end
        
        normp = norm(p,Inf)

        # 2) Evaluate how bad is the model step compared to real function step
        ρ = (F(x+p)[1] - F(x)[1])/( 0.5*p'*B*p + dot(g,p) )

        # 3) Δk uptade based on ρ
        if (ρ < η)
            Δk = η*Δk
            #Δk = η*normp*ones(l)
            # Δk = η*abs.(p)
        else
            if (ρ > η2 && τ == 1)
                Δk = min(2*maximum(Δk), Δ)*ones(l,2)
            else #se x saiu da restrição, então aumentar o delta_k naquela direção
                Δk = maximum(Δk)*ones(l,2)
            end
        end
        
        # 4)Acceptance of step
        if (ρ > η)
            x += p
        end
        
        k += 1
        
        if ( normp<ϵ )
            break
        end
        
    end
    @printf("O número de iterações é = ")
    println(k)
    @printf("O valor de x* final é = ")
    println(x)
    @printf("O valor de f(x*) final é = ")
    println(F(x))
    @printf("O valor de ∇f(x*) final é = ")
    println(g)
    opt_tester = opt_test(x,g,λ,lo,up)
    @printf("O valor de λ* final é = ")
    println(λ)
    @printf("O teste de otimo deu (1-fail, 0-succes) = ")
    println(opt_tester)
    @printf("O norm(p) = ")
    println(normp)
end

function opt_test(x,gg,λ,lo,up; tol=1e-4)
    l = length(x)
    g = copy(gg)

    #Does LICQ hold?
    #Always hold for box-constrained optimization

    #Is x feasible?
    if ( minimum(x-lo)<0 ||  minimum(up-x)<0 )
        println("KKT falhou por infeasibility")
        return 1
    end
    
    #Is there some λ* such that ∇L(x*,λ*) = 0?
    A = zeros(l,2*l)
    for i=1:l
        A[i,2*i-1] = 1
        A[i,2*i] = -1
    end
    for i=1:l
        if ( abs(x[i]-lo[i]) > 0.001 && abs(up[i]-x[i]) > 0.001 )
            tmp = zeros(1,2*l)
            tmp[2*i-1] = 1
            tmp[2*i] = 1
            A = cat(A, tmp, dims = 1)
            g = [g;0]
        else
            if ( abs(x[i]-lo[i]) > 0.001 )
                tmp = zeros(1,2*l)
                tmp[2*i-1] = 1
                A = cat(A, tmp, dims = 1)
                g = [g;0]
            end
            if (abs(up[i]-x[i]) > 0.001)
                tmp = zeros(1,2*l)
                tmp[2*i] = 1
                A = cat(A, tmp, dims = 1)
                g = [g;0]
            end
        end
    end
    λ .= A\g

    #   λ* > 0?
    if (minimum(λ)>=-tol)
        #println("quase otimo")
        return 0
    else
        return 1
    end
end

function plot_3d(F;l=[0.,0.],u=[10.,10.])
    #l = length(x1)
    r = 100
    x1 = l[1]:(u[1]-l[1])/(r-1):u[1]
    x2 = l[2]:(u[2]-l[2])/(r-1):u[2]
    #y  = zeros(l,l)
    y  = zeros(r,r)
    for i=1:r
        for j=1:r
            y[i,j] = F([x1[j], x2[i]])[1]
        end
    end
    
    #plot(x1,x2,y,st=:surface,camera=(30,30), xlabel = "x1", ylabel = "x2")
    PyPlot.contour(x1,x2,y, xlabel="x1", ylabel="x2", 20)

    #p2 = surface(x1,x2,y,xlabel="x1", ylabel="x2")
    #plot(p1,p2)
    #plot(p1)
    #plot(x1,x2,y,st=:contour,xlabel = "x1", ylabel = "x2")
    
end

function Fbranin(x)
    l = length(x[1,:])

    return (x[2,:] - 0.129*x[1,:].^2 + 1.6*x[1,:] - 6*ones(l)).^2 +6.07*cos.(x[1,:]) + 10*ones(l)
end

function Gbranin!(x,grad)
    grad[1] = 2*(x[2] - 0.129*x[1]^2 + 1.6*x[1] - 6)*(-2*0.129*x[1] + 1.6) - 6.07*sin(x[1])
    grad[2] = 2*(x[2] - 0.129*x[1]^2 + 1.6*x[1] - 6)
end

function Hbranin!(x,H)
    H[1,1] = 2*( (-2*0.129*x[1] + 1.6)^2 -2*0.129*( x[2] - 0.129*x[1]^2 + 1.6*x[1] - 6) ) - 6.07*cos(x[1])
    H[2,1] = H[1,2] = 2*(-2*0.129*x[1] + 1.6)
    H[2,2] = 2
end

function callbranin(;fl=0, t="ro")
    TR_box!(Fbranin, Gbranin!, Hbranin!, [10.,10.], [0.,0.], [10.,15.], flag=fl, Δ=4, ITMAX=10000, ttt=t)
    #TR_box!(Fbranin, Gbranin!, Hbranin!, [6.,14.], [3.,3.], [14.,14.], flag=-1 , Δ=4, ITMAX=1000)
    #TR_box!(Fbranin, Gbranin!, Hbranin!, [6.,14.], [6.,3.], [14.,14.], flag=-1 , Δ=4, ITMAX=1000)
    #TR_box!(Fbranin, Gbranin!, Hbranin!, [6.,14.], [0.,0.], [6.,14.], flag=-1 , Δ=4, ITMAX=1000)
end

function F110(x)
    l = 10
    return dot(log.(x-2*ones(l)).^2 + log.(10*ones(l) - x).^2, ones(l)) - prod(x)^.2
end

function G110!(x,grad)
    prodx = prod(x)
    j = 1
    for i in x
        grad[j] = 2*log(i-2)/(i-2) - 2*log(10-i)/(10-i) - 0.2*(prodx^0.2)/i
        j+=1
    end
end

function H110!(x,H)
    l = 10
    prodx = prod(x.^0.2)
    for i=1:l
        for j=i:l
            if (i==j)
                H[i,j] = 2*(1-log(x[i]-2))/(x[i]-2)^2 - 2*(log(x[i]-2) - 1)/(10 - x[i])^2 + 0.16*prodx/x[i]^2
            else
                H[i,j] = H[j,i] = -0.04*prodx/(x[j]*x[i].+eps())
            end
        end
    end

end

function call110(;fl = 0, delta = 3)
    TR_box!(F110, G110!, H110!, 9*ones(10), 2.001*ones(10), 9.999*ones(10), flag=fl, Δ=delta, ITMAX=10000)
end

function F45(x)
    return 2 - prod(x)/120
end

function G45!(x,grad)
    grad .= (-1/120)*prod(x)./(x .+ eps())
end

function H45!(x,H)
    l = 5
    prodx = prod(x)
    for i=1:l
        for j=i:l
            if (i==j)
                H[i,j] = 0
            else
                H[i,j] = H[j,i] = -prodx/(x[i]*x[j]*120 .+ eps())
            end
        end
    end

end

function call45(;fl = 0, delta = 3)
    #em realidade o teste começa infeasible, mas o algoritmo não da conta
    TR_box!(F45, G45!, H45!, 2*ones(5), zeros(5), [1.,2.,3.,4.,5.], flag=fl, Δ=delta, ITMAX=1000)
    #TR_box!(F45, G45!, H45!, 0.5*ones(5), zeros(5), [1.,2.,3.,4.,5.], flag=-1, Δ=3, ITMAX=1000)
end

function F38(x)
    return 100*(x[2]-x[1]^2)^2 + (1-x[1])^2 + 90*(x[4]-x[3]^2)^2 + (1-x[3])^2 + 10.1*( (x[2]-1)^2 + (x[4]-1)^2 ) + 19.8*(x[2]-1)*(x[4]-1)
end

function G38!(x,grad)
    grad[1] = -400*( x[2]-x[1]^2 )*x[1] - 2 + 2*x[1]
    grad[2] =  200*( x[2]-x[1]^2 ) + 20.2*(x[2] - 1) + 19.8*(x[4]-1)
    grad[3] = -360*( x[4]-x[3]^2 )*x[3] - 2 + 2*x[3]
    grad[4] =  180*( x[4]-x[3]^2 ) + 20.2*(x[4] - 1) + 19.8*(x[2]-1)
end

function H38!(x,H)
    H[1,1] = -400*(-3*x[1]^2 + x[2]) + 2
    H[2,2] = 220.2
    H[3,3] = -360*(-3*x[3]^2 + x[4]) + 2
    H[4,4] = 200.2
    H[1,2] = H[2,1] = -400*x[1]
    H[1,3] = H[3,1] = 0
    H[1,4] = H[4,1] = 0
    H[2,3] = H[3,2] = 0
    H[2,4] = H[4,2] = 19.8
    H[3,4] = H[4,3] = -360*x[3]
end

function call38(;fl = 0, delta = 3)
    #TR_box!(F38, G38!, H38!, [-3., -1., -3., -1.], -10*ones(4), 10*ones(4), flag=0, Δ=0.1, ITMAX=1000)
    TR_box!(F38, G38!, H38!, [3., -1., -3., -1.], -10*ones(4), 10*ones(4), flag=fl, Δ=delta, ITMAX=10000, ϵ = 1e-6)
    #converge para ótimo local, mas critério de otimilidade não detecta
    #talvez o método não apresente convergencia rápida perto da solução
    #∇f(x) muda muito rápido perto da solução
end

function F5(x)
    return sin(x[1]+x[2]) + (x[1]-x[2])^2 - 1.5*x[1] +2.5*x[2] + 1
end

function G5!(x,grad)
    grad[1] = cos(x[1]+x[2]) + 2*(x[1]-x[2]) - 1.5
    grad[2] = cos(x[1]+x[2]) - 2*(x[1]-x[2]) + 2.5
end

function H5!(x,H)
    H .= -sin(x[1]+x[2])*ones(2,2)
    H[1,1] += 2
    H[2,2] += 2
    H[2,1] += -2
    H[1,2] += -2

end

function call5(;fl = 0, delta = 3, t="ro")
    #TR_box!(F38, G38!, H38!, [-3., -1., -3., -1.], -10*ones(4), 10*ones(4), flag=0, Δ=0.1, ITMAX=1000)
    TR_box!(F5, G5!, H5!, [0., 0.], [-0.5,-1.], [4.,3.], flag=fl, Δ=delta, ITMAX=10000, ϵ = 1e-6, ttt=t)

end

function F4(x)
    return (x[1]+1)^3/3+x[2]
end

function G4!(x,grad)
    grad[1] = (x[1]+1)^2
    grad[2] = 1
end

function H4!(x,H)
    H .= zeros(2,2)
    H[1,1] += 2*x[1]+2
end

function call4(;fl = 0, delta = 3)
    #TR_box!(F38, G38!, H38!, [-3., -1., -3., -1.], -10*ones(4), 10*ones(4), flag=0, Δ=0.1, ITMAX=1000)
    TR_box!(F4, G4!, H4!, [1.125, 0.125], [1,0], [Inf,Inf], flag=fl, Δ=delta, ITMAX=10000, ϵ = 1e-6)
    plot_3d(F4, l=[1.,0.], u=[5.,4.])
end

function F3(x)
    return 1e-5*(x[1]-x[2])^2 + x[2]
end

function G3!(x,grad)
    grad[1] = 2e-5*(x[1]-x[2])
    grad[2] = 1 - 2e-5*(x[1]-x[2])
end

function H3!(x,H)
    H[1,1] = 2e-5
    H[1,2] = -2e-5
    H[2,1] = -2e-5
    H[2,2] = 2e-5
end

function call3(;fl = 0, delta = 3)
    TR_box!(F3, G3!, H3!, [10., 1.], [-Inf,0], [Inf,Inf], flag=fl, Δ=delta, ITMAX=50, ϵ = 1e-6)
    #TR_box!(F3, G3!, H3!, [10., 1.], [-Inf,0], [Inf,Inf], flag=-1, Δ=5, ITMAX=10000, ϵ = 1e-6)
    #funciona mal com projeção pura, pois o gradiente é muito pequeno, então o passo também é...
    plot_3d(F3, l=[-10.,0.], u=[10.,10.])
end

#Flags:
# -1: TR step uses projection only
#  0: TR step uses Cauchy point IN the TR and projection AT TR

#problemas
#-> cálculo de p_cauchy quando x[i]==(up[i] or lo[i])
#-> cálculo de p_projec quando x[i]==(up[i] or lo[i]) - garantir decreasing direction (ex.: baixo delta)
#-> steinhaugh's não funcionou
#-> pensar nas atualizações de Δk no algoritmo de TR

#O que eu tenho:
#Algoritmo de região de confiança;
#Cálculo do passo por p_cauchy caso x[i]!=(up[i] or lo[i])
#Cálculo do passo por projeção do -g na região de confiança com line search (backcalculation)
#Cálculo do passo por projeção do -g na região de confiança com solução explicita do modelo quadratico
