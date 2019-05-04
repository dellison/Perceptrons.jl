@testset "Binary Averaged Perceptrons" begin

    p = AveragedPerceptron(5)

    x, y = [0,0,0,0,0], true
    @test p.p.b == 0
    @test p.p.w == zeros(5)
    @test score(p, x) == 0
    @test predict(p, x) == false
    Perceptrons.fit_one!(p, x, y)

    x, y = [1,0,0,0,0], false
    @test p.p.b == 1
    @test p.p.w == [0,0,0,0,0]
    @test score(p, x) == 1
    @test predict(p, x) == true
    Perceptrons.fit_one!(p, x, y)
    @test p.p.b == 0
    @test p.p.w == [-1,0,0,0,0]
    @test score(p, x) == -1

    x, y = [1,2,0,1,0], false
    @test score(p, x) == -1
    @test predict(p, x) == false
end

@testset "Multiclass Averaged Perceptrons" begin

end
