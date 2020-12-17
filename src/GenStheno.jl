module GenStheno

using Gen
using Stheno
using Zygote
using MacroTools
using MacroTools: @capture

# ------------ Choice map ------------ #

struct SthenoChoiceMap <: ChoiceMap
    val::IdDict
end

@inline Gen.get_values_shallow(chm::SthenoChoiceMap) = chm.val
@inline Gen.get_submaps_shallow(chm::SthenoChoiceMap) = ()

# ------------ Trace ------------ #

struct SthenoTrace{N} <: Gen.Trace
    score::Float64
    gen_fn::GenerativeFunction
    args::Tuple
    retval::Array{Float64, N}
end

@inline Gen.get_args(trace::SthenoTrace) = trace.args
@inline Gen.get_retval(trace::SthenoTrace) = trace.retval
@inline Gen.get_score(trace::SthenoTrace) = trace.score
@inline function Gen.get_choices(trace::SthenoTrace)
    x, y = get_args(trace)[1], get_retval(trace)
    SthenoChoiceMap(IdDict([zip(x, y)...]))
end
@inline Gen.get_gen_fn(trace::SthenoTrace) = trace.gen_fn

# ------------ Generative function ------------ #

struct SthenoGenerativeFunction <: Gen.GenerativeFunction{Array{Float64}, SthenoTrace}
    generator::Function
    σ²_n::Float64 # iid observation noise
    ppr_n::Float64 # posterior predictive noise
    SthenoGenerativeFunction(fn, σ²_n) = new(fn, σ²_n, 0.0)
end

@inline (sf::SthenoGenerativeFunction)(args...) = Gen.get_retval(Gen.simulate(sf, args))

# ------------ GFI ------------ #

function Gen.simulate(gen_fn::SthenoGenerativeFunction, args::Tuple)
    x, params = args[1], args[2 : end]
    gp = gen_fn.generator(params...)
    @assert gp isa Stheno.GP
    fx = gp(x, gen_fn.σ²_n)
    s = rand(fx)
    score = Stheno.logpdf(fx, s)
    SthenoTrace(score, gen_fn, args, s)
end

function Gen.generate(gen_fn::SthenoGenerativeFunction, args::Tuple, chm::SthenoChoiceMap)
    x, params = args[1], args[2 : end]
    gp = gen_fn.generator(params...)
    @assert gp isa Stheno.GP
    constrain_x, constrain_y = collect(keys(chm.val)), collect(values(chm.val))
    fx = gp(constrain_x, gen_fn.σ²_n)
    f_post = gp | Obs(fx, constrain_y) # condition on choice map
    f_post_on_x = f_post(x, gen_fn.ppr_n)
    pred = rand(f_post_on_x) # predictive posterior on x
    weight = Stheno.logpdf(fx, constrain_y)
    score = Stheno.logpdf(fx, pred)
    SthenoTrace(score, gen_fn, args, pred), weight
end

function Gen.propose(gen_fn::SthenoGenerativeFunction, args::Tuple)
    trace = simulate(gen_fn, args)
    retval = get_retval(trace)
    (SthenoChoiceMap(retval), get_score(trace), retval)
end

function Gen.update(trace::SthenoTrace, args::Tuple, argdiffs::Tuple, chm::SthenoChoiceMap)
    new_trace, w = generate(trace.gen_fn, args, chm)
    (new_trace, w, UnknownChange(), get_choices(trace))
end

function Gen.update(trace::SthenoTrace, args::Tuple, argdiffs::Tuple, chm::EmptyChoiceMap)
    all(x -> x isa NoChange, argdiffs) && return (trace, 0., NoChange(), EmptyChoiceMap())
    new_trace = simulate(trace.gen_fn, args)
    (new_trace, get_score(trace) - get_score(new_trace), UnknownChange(), EmptyChoiceMap())
end

function Gen.regenerate(trace::SthenoTrace, args::Tuple, argdiffs::Tuple,::EmptySelection)
    all(x -> x isa NoChange, argdiffs) && return (trace, 0.0, NoChange())
    new_trace = simulate(trace.gen_fn, args)
    (new_tr, get_score(new_trace) - get_score(trace), UnknownChange())
end

function Gen.regenerate(trace::SthenoTrace, args::Tuple, argdiffs::Tuple,::Gen.AllSelection)
    new_trace = simulate(trace.gen_fn, args)
    (new_tr, get_score(new_trace) - get_score(trace), UnknownChange())
end

# ------------ Macro ------------ #

function _genstheno(expr)
    @capture(expr, (generator = fn_, σ = val_))
    Expr(:call, GlobalRef(GenStheno, :SthenoGenerativeFunction), fn, val)
end

macro sthen(expr)
    new = _genstheno(expr)
    esc(new)
end

export @sthen, SthenoChoiceMap

end # module
