type Vertex
    ID::UInt32
    rank::Float64
    activated::Bool
    in_nbrs::Vector{UInt32}
    out_nbrs::Vector{UInt32}
    delta::Float64
end

gather(vertex::Vertex)::Vector{Float64} = [vertexes[in_nbr].rank / length(vertexes[in_nbr].out_nbrs) for in_nbr in vertex.in_nbrs]
function apply(vertex::Vertex, acc::Float64)
    new_rank = 0.15 + 0.85 * acc
    vertex.delta = (new_rank - vertex.rank) / length(vertex.out_nbrs)
    vertex.rank = new_rank
    vertex.activated = false
end
function scatter(vertex::Vertex, delta::Float64)::Vector{UInt32}
    if abs(vertex.delta) > delta
        return vertex.out_nbrs
    else
        return Vector{UInt32}()
    end
end

delta = 0.01
data_path = "./rank.txt"
vertexes = Dict{UInt32, Vertex}()
# Build Graph
function add_vertex(vertexes::Dict{UInt32, Vertex}, out_node::UInt32, in_node::UInt32)
    vertex = get(vertexes, out_node, 0)
    if vertex == 0
        in_nbrs = Vector{UInt32}()
        push!(in_nbrs, in_node)
        vertexes[out_node] = Vertex(out_node, 1.0::Float64, true, in_nbrs, Vector{UInt32}(), 0.0::Float64)
    else
        push!(vertex.in_nbrs, in_node)
    end
    vertex = get(vertexes, in_node, 0)
    if vertex == 0
        out_nbrs = Vector{UInt32}()
        push!(out_nbrs, out_node)
        vertexes[in_node] = Vertex(in_node, 1.0::Float64, true, Vector{UInt32}(), out_nbrs, 0.0::Float64)
    else
        push!(vertex.out_nbrs, out_node)
    end
end

open(data_path) do edges
    for line in eachline(edges)
        if line != ""
            edge = split(strip(line),  "\t")
            add_vertex(vertexes, parse(UInt32, edge[1]), parse(UInt32, edge[2]))
        end
    end
end

# Begin Serial PageRank
while true
    flag = false
    for vertex in vertexes
        vertex = vertex[2]
        if vertex.activated
            accs = gather(vertex)
            apply(vertex, sum(accs))
            act_nbrs = scatter(vertex, delta)
            for nbr in act_nbrs
                vertexes[nbr].activated = true
                flag = true
            end
        end
    end
    if flag == false
        break
    end
end

# Print Results
for vertex in vertexes
    println("User $(vertex[1]) has rank $(vertex[2].rank)")
end
