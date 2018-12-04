@testset "Binary Perceptrons" begin
    using Perceptrons: BinaryPerceptron
    using Perceptrons: fit_one!, predict, score, update!, weight

    p = BinaryPerceptron(Dict)
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

    phi = [1, 2, 3]

    p = BinaryPerceptron(5)
    @test score(p, phi) == 0
    fit_one!(p, (1,2,3), true)
    fit_one!(p, (3,4,5), false)
    @test predict(p, [1]) == predict(p, [2])  == true
    @test predict(p, [3]) == predict(p, [4]) == predict(p, [5]) == false
end
