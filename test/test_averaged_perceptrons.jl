@testset "Binary Averaged Perceptrons" begin

    using Perceptrons: BinaryAveragedPerceptron
    using Perceptrons: fit_one!, score, predict

    p = BinaryAveragedPerceptron()
    emails = [
        (split("meeting today"), false),
        (split("free money today"), true)
    ]
    for (email, label) in emails
        fit_one!(p, email, label)
    end
    is_spam(str) = predict(p, split(str))
    @test is_spam("free money tomorrow")
    @test !is_spam("meeting tomorrow")

    p = BinaryAveragedPerceptron(5)
    phi = [1, 2, 3]

    @test score(p, phi) == 0
    fit_one!(p, (1,2,3), true)
    fit_one!(p, (3,4,5), false)
    @test predict(p, [1]) == predict(p, [2]) == true
    @test predict(p, [3]) == predict(p, [4]) == predict(p, [5]) == false
end

@testset "Averaged Perceptrons" begin
    using Perceptrons: AveragedPerceptron

    p = AveragedPerceptron(Dict, ("A", "B", "C"))

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

    using Perceptrons: train!
    xs, ys = zip((f("that was good"), "pos"),
                 (f("that was bad"), "neg"),
                 (f("that was so-so"), "neu"))
    p = AveragedPerceptron(Dict, unique(ys))
    train!(p, xs, ys)
    @test predict(p, f("good")) == "pos"
    @test predict(p, f("bad")) == "neg"
    @test predict(p, f("so-so")) == "neu"
end
