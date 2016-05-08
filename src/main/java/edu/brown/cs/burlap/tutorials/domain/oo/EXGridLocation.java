package edu.brown.cs.burlap.tutorials.domain.oo;

import burlap.mdp.core.oo.state.ObjectInstance;

import static edu.brown.cs.burlap.tutorials.domain.oo.ExampleOOGridWorld.CLASS_LOCATION;

/**
 * @author James MacGlashan.
 */
public class EXGridLocation extends ExGridAgent{

	public int type;

	public EXGridLocation() {
	}


	public EXGridLocation(int x, int y, String name) {
		super(x, y, name);
	}

	public EXGridLocation(int x, int y, int type, String name) {
		super(x, y, name);
		this.type = type;
	}

	@Override
	public String className() {
		return CLASS_LOCATION;
	}

	@Override
	public ObjectInstance copyWithName(String objectName) {
		return new EXGridLocation(x, y, type, objectName);
	}

	@Override
	public EXGridLocation copy() {
		return new EXGridLocation(x, y, type, name);
	}
}
