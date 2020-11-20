# Convert piece-wise constant p (design region) to pvec (whole domain)
function p_vec(p,P,tags,design_tag)
    pvec = zeros(num_free_dofs(P))
    pi = 0
    @assert length(tags)==num_free_dofs(P)
    for i=1:length(tags)
        if tags[i] == design_tag
            pi += 1
            pvec[i] = p[pi]
        end
    end
    pvec
end

# Extract the design region part from a whole vector
function extract_design(pvec,np,tags,design_tag)
    p_d = zeros(eltype(pvec),np)
    pi = 0
    @assert length(pvec)==length(tags)
    for i=1:length(tags)
        if tags[i] == design_tag
            pi += 1
            p_d[pi] = pvec[i]
        end
    end
    @assert np==pi
    p_d
end

