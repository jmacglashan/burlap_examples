package edu.brown.cs.burlap.tutorials.domain.oo;

import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.StateUtilities;
import burlap.mdp.core.state.UnknownKeyException;
import burlap.mdp.core.state.annotations.DeepCopyState;

import java.util.Arrays;
import java.util.List;

import static edu.brown.cs.burlap.tutorials.domain.oo.ExampleOOGridWorld.CLASS_AGENT;
import static edu.brown.cs.burlap.tutorials.domain.oo.ExampleOOGridWorld.VAR_X;
import static edu.brown.cs.burlap.tutorials.domain.oo.ExampleOOGridWorld.VAR_Y;


/**
 * @author James MacGlashan.
 */
@DeepCopyState
public class ExGridAgent implements ObjectInstance, MutableState {

	public int x;
	public int y;

	public String name = "agent";

	private final static List<Object> keys = Arrays.<Object>asList(VAR_X, VAR_Y);

	public ExGridAgent() {
	}

	public ExGridAgent(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public ExGridAgent(int x, int y, String name) {
		this.x = x;
		this.y = y;
		this.name = name;
	}

	@Override
	public String className() {
		return CLASS_AGENT;
	}

	@Override
	public String name() {
		return name;
	}

	@Override
	public ObjectInstance copyWithName(String objectName) {
		return new ExGridAgent(x, y, objectName);
	}

	@Override
	public MutableState set(Object variableKey, Object value) {
		if(variableKey.equals(VAR_X)){
			this.x = StateUtilities.stringOrNumber(value).intValue();
		}
		else if(variableKey.equals(VAR_Y)){
			this.y = StateUtilities.stringOrNumber(value).intValue();
		}
		else{
			throw new UnknownKeyException(variableKey);
		}
		return this;
	}

	@Override
	public List<Object> variableKeys() {
		return keys;
	}

	@Override
	public Object get(Object variableKey) {
		if(variableKey.equals(VAR_X)){
			return x;
		}
		else if(variableKey.equals(VAR_Y)){
			return y;
		}
		throw new UnknownKeyException(variableKey);
	}

	@Override
	public ExGridAgent copy() {
		return new ExGridAgent(x, y, name);
	}

	@Override
	public String toString() {
		return StateUtilities.stateToString(this);
	}

}
