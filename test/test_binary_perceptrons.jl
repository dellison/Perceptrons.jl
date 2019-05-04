@testset "Binary Perceptrons" begin
    @testset "Perceptrons" begin

        p = Perceptron(5)

        x, y = [0,0,0,0,0], true
        @test p.b == 0
        @test p.w == [0,0,0,0,0]
        @test score(p, x) == 0
        @test predict(p, x) == false
        update!(p, x, true)

        @test p.b == 1
        @test p.w == [0,0,0,0,0]
        @test score(p,   [1,0,0,0,0]) == 1
        @test predict(p, [1,0,0,0,0]) == true

        x, y = [1,2,0,0,0], false
        @test score(p, x) == 1
        @test predict(p, x) == true != y
        update!(p, x, y)
        @test p.w == [-1,-2,0,0,0]
        @test p.b == 0
    end
end
