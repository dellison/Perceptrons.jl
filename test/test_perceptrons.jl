@testset "Perceptrons" begin
    using Perceptrons: Perceptron
    using Perceptrons: fit_one!, predict, score, scores, update!, weight

    p = Perceptron(Dict, ("A", "B", "C"))

    @test all(score(p, y, ("f1", "f2", "f3")) == 0 for y in ("A", "B", "C"))

    fit_one!(p, "A", ("f1", "f2", "f3"))
    fit_one!(p, "B", ("f3", "f4", "f5"))

    @test predict(p, ("f1",)) == "A"
    @test predict(p, ("f4",)) == "B"

    p = Perceptron(Dict, ("pos", "neg", "neu"))
    f(str) = split(str)
    fit_one!(p, "pos", f("that was good"))
    fit_one!(p, "neg", f("that was bad"))
    fit_one!(p, "neu", f("that was so-so"))
    @test predict(p, f("good")) == "pos"
    @test predict(p, f("bad")) == "neg"
    @test predict(p, f("so-so")) == "neu"
end
