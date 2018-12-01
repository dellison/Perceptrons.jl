@testset "Binary Averaged Perceptrons" begin

    using Perceptrons: BinaryAveragedPerceptron
    using Perceptrons: fit_one!, score, predict

    p = BinaryAveragedPerceptron()
    emails = [
        (split("meeting today"), false),
        (split("free money today"), true)
    ]
    for (email, label) in emails
        fit_one!(p, label, email)
    end
    is_spam(str) = predict(p, split(str))
    @test is_spam("free money tomorrow")
    @test !is_spam("meeting tomorrow")

    p = BinaryAveragedPerceptron(5)
    phi = [1, 2, 3]

    @test score(p, phi) == 0
    fit_one!(p, true, (1,2,3))
    fit_one!(p, false, (3,4,5))
    @test predict(p, [1]) == predict(p, [2]) == true
    @test predict(p, [3]) == predict(p, [4]) == predict(p, [5]) == false
end

@testset "Averaged Perceptrons" begin
    using Perceptrons: AveragedPerceptron

    p = AveragedPerceptron(Dict, ("A", "B", "C"))

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

    using Perceptrons: train!
    Xs, Ys = zip((f("that was good"), "pos"),
                 (f("that was bad"), "neg"),
                 (f("that was so-so"), "neu"))
    p = AveragedPerceptron(Dict, unique(Ys))
    train!(p, Xs, Ys)
    @test predict(p, f("good")) == "pos"
    @test predict(p, f("bad")) == "neg"
    @test predict(p, f("so-so")) == "neu"
end
