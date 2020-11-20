@law myabs2(x) = abs2(x)

#g=g_u(uvec)
function g_u(uvec)
    uh_temp = FEFunction(U,uvec)
    uh_t_temp = restrict(uh_temp,trian_t)
    sum(integrate(myabs2(uh_t_temp),trian_t,quad_t))
end
#uvec = u_pf(pf)
function u_pf(pf)
    A,b = GridapFEM(pf,b_Ω,α,β,η,k,σ,LH,d_pml)
    uvec = A\b
    uvec
end
#pf = pf_p(p)
function pf_p(p)
    pf = Filter(p,r,P,Pf,Qf,tags,design_tag,trian,quad,btrian,bquad)
    pf
end
# Chain Rule : dg/dp = dg/dg*dg/du*du/dpf*dpf/dp
# dg/du=dg/dg*dg/du
function rrule(::typeof(g_u),uvec)
  function g_pullback(dgdg)
    NO_FIELDS, dgdg*Dgdu(uvec)
  end
  g_u(uvec), g_pullback
end

function Dgdu(uvec)
  uh = FEFunction(U,uvec)
  uh_t = restrict(uh,trian_t)
  t = FESource(du->(2*uh_t*du),trian_t,quad_t)
  op = AffineFEOperator(SparseMatrixCSC{ComplexF64,Int},Vector{ComplexF64},U,V,t)
  get_vector(op)
end

# dg/dpf=dg/du*du/dpf
function rrule(::typeof(u_pf),pf)
  uvec = u_pf(pf)
  function u_pullback(dgdu)
    NO_FIELDS, Dgdpf(dgdu,uvec,pf)
  end
  uvec, u_pullback
end

@law function Dxidpf(pfh,tag)
    if tag == design_tag
        return (ϵ_1-ϵ_2+α*1im*(1-2*Threshold(β,η,pfh)))/ξ(Threshold(β,η,pfh),α)^2*β*(1.0-operate(tanh,β*(pfh-η))^2)/(tanh(β*η)+tanh(β*(1.0-η)))
    else
        return 0.0+0im
    end
end

function Dgdpf(dgdu,uvec,pf)

  da(pfh,u,v,dp) = -Dxidpf(pfh,tags)*∇(v)⊙∇(u)*dp

  A,b = GridapFEM(pf,b_Ω,α,β,η,k,σ,LH,d_pml)
  λvec = A'\dgdu
  
  uh = FEFunction(U,uvec)
  ph = FEFunction(Pf,pf)
  λh = FEFunction(V,conj(λvec))

  t = FESource((dp)->da(ph,uh,λh,dp),trian,quad)
  op = AffineFEOperator(SparseMatrixCSC{ComplexF64,Int},Vector{ComplexF64},Pf,Qf,t)
  real(get_vector(op))
end

# dg/dp=dg/dpf*dpf/dp
function rrule(::typeof(pf_p),p)
  function pf_pullback(dgdpf)
    NO_FIELDS, Dgdp(dgdpf)
  end
  Filter(p,r,P,Pf,Qf,tags,design_tag,trian,quad,btrian,bquad), pf_pullback
end

function Dgdp(dgdpf)
  t_Ω = LinearFETerm((u,v)->a_f(r,u,v),trian,quad)
  op = AffineFEOperator(Pf,Qf,t_Ω)
  A = get_matrix(op)
  λvec = A'\dgdpf

  λh = FEFunction(Pf,λvec)
  t = FESource(dp->(λh*dp),trian,quad)
  op = AffineFEOperator(P,Q,t)
  extract_design(get_vector(op),np,tags,design_tag)
end


# Final objective function
function g_p(p::Vector)
    #g(u_pt(Threshold(Filter(p))))
    pf = pf_p(p)
    uvec = u_pf(pf)
    g_u(uvec)
end

function g_p(p::Vector,grad::Vector)
    if length(grad) > 0
        dgdp, = Zygote.gradient(g_p,p)
        grad[:] = dgdp
    end
    @show g_value = g_p(p)
    return g_value
end