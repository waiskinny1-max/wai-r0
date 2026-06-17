from wai_r0.symbolic import Grid, rotate90, mirror_y, crop_nonzero, fill_enclosed, ProgramSearch, ArcTask, TaskExample

def test_dsl_ops():
    assert rotate90(Grid.from_lists([[1,2],[3,4]])).to_lists()==[[3,1],[4,2]]
    assert mirror_y(Grid.from_lists([[1,2,3]])).to_lists()==[[3,2,1]]
    assert crop_nonzero(Grid.from_lists([[0,0,0],[0,2,0],[0,0,0]])).to_lists()==[[2]]
    assert fill_enclosed(Grid.from_lists([[1,1,1],[1,0,1],[1,1,1]])).to_lists()==[[1,1,1],[1,1,1],[1,1,1]]

def test_program_search():
    task=ArcTask('r90',(TaskExample(Grid.from_lists([[1,2],[3,4]]),Grid.from_lists([[3,1],[4,2]])),),(TaskExample(Grid.from_lists([[7,8],[9,1]]),None),))
    r=ProgramSearch(max_depth=1).solve(task); assert r.solved; assert r.program=='rotate90'; assert r.predictions==[[[9,7],[1,8]]]
