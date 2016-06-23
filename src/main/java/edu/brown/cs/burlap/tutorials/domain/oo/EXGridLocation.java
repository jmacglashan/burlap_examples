package edu.brown.cs.burlap.tutorials.domain.oo;

import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.StateUtilities;

import java.util.Arrays;
import java.util.List;

import static edu.brown.cs.burlap.tutorials.domain.oo.ExampleOOGridWorld.*;

/**
 * @author James MacGlashan.
 */
public class EXGridLocation extends ExGridAgent{

	public int type;

	private final static List<Object> keys = Arrays.<Object>asList(VAR_X, VAR_Y, VAR_TYPE);

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
	public List<Object> variableKeys() {
		return keys;
	}

	@Override
	public Object get(Object variableKey) {
		if(variableKey.equals(VAR_TYPE)){
			return this.type;
		}
		return super.get(variableKey);
	}

	@Override
	public MutableState set(Object variableKey, Object value) {
		if(variableKey.equals(VAR_TYPE)){
			this.type = StateUtilities.stringOrNumber(value).intValue();
		}
		else{
			super.set(variableKey, value);
		}
		return this;

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
