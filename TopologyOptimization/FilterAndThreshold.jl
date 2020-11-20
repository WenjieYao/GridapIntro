################### Filter and Threshold #####################
# pf = Filter(p)
a_f(r,u,v) = r^2*(∇(v)⊙∇(u))+v⊙u
function Filter(p,r,P,Pf,Qf,tags,design_tag,trian,quad,btrian,bquad)
    pvec = p_vec(p,P,tags,design_tag)
    ph = FEFunction(P,pvec)
    t_Ω = AffineFETerm((u,v)->a_f(r,u,v),v->(v*ph),trian,quad)
    t_Γ = FESource(v->(0.0*v),btrian,bquad)
    op = AffineFEOperator(Pf,Qf,t_Ω,t_Γ)
    pfh = solve(op)
    get_free_values(pfh)
end

# Threshold function
@law Threshold(β,η,pf) = (tanh(β*η)+operate(tanh,β*(pf-η)))/(tanh(β*η)+tanh(β*(1.0-η)))