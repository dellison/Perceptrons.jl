@testset "Perceptrons" begin
    using Perceptrons: Perceptron
    using Perceptrons: fit_one!, predict, score, scores, update!, weight

    p = Perceptron(Dict, ("A", "B", "C"))

    @test all(score(p, ("f1", "f2", "f3"), y) == 0 for y in ("A", "B", "C"))

    fit_one!(p, ("f1", "f2", "f3"), "A")
    fit_one!(p, ("f3", "f4", "f5"), "B")

    @test predict(p, ("f1",)) == "A"
    @test predict(p, ("f4",)) == "B"

    p = Perceptron(Dict, ("pos", "neg", "neu"))
    f(str) = split(str)
    fit_one!(p, f("that was good"), "pos")
    fit_one!(p, f("that was bad"), "neg")
    fit_one!(p, f("that was so-so"), "neu")
    @test predict(p, f("good")) == "pos"
    @test predict(p, f("bad")) == "neg"
    @test predict(p, f("so-so")) == "neu"
end
