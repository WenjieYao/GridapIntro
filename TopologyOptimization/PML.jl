# PML coordinate streching functions
function s_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2
    return @. ifelse(u > 0,  1-(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

function Λ(x,σ,k,LH,d_pml)  
    s_x,s_y = s_PML(x,σ,k,LH,d_pml)
    return TensorValue(1/s_x,0,0,1/s_y) 
end