
/**
 * Object manager component for creating and managing annotation objects
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Plus, Trash2, Circle, Target, Palette } from 'lucide-react';
import { HexColorPicker } from 'react-colorful';
import { AnnotationObject } from '@/types/annotation';

interface ObjectManagerProps {
  objects: AnnotationObject[];
  selectedObject: string | null;
  onObjectsChange: (objects: AnnotationObject[]) => void;
  onObjectSelect: (objectId: string | null) => void;
  annotationMode: 'positive' | 'negative';
  onAnnotationModeChange: (mode: 'positive' | 'negative') => void;
  onObjectDeleted?: (objectId: string) => void;
}

/**
 * Generate a random color that's not already in use
 */
const generateUniqueColor = (usedColors: string[]): string => {
  const colors = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', 
    '#8b5cf6', '#06b6d4', '#f97316', '#84cc16',
    '#ec4899', '#14b8a6', '#f43f5e', '#10b981',
    '#6366f1', '#f59e0b', '#8b5cf6', '#06b6d4'
  ];
  
  const availableColors = colors.filter(color => !usedColors.includes(color));
  if (availableColors.length > 0) {
    return availableColors[0];
  }
  
  // Generate random color if all presets are used
  return `#${Math.floor(Math.random()*16777215).toString(16).padStart(6, '0')}`;
};

/**
 * Component for managing annotation objects and modes
 */
export const ObjectManager: React.FC<ObjectManagerProps> = ({
  objects,
  selectedObject,
  onObjectsChange,
  onObjectSelect,
  annotationMode,
  onAnnotationModeChange,
  onObjectDeleted
}) => {
  const [newObjectName, setNewObjectName] = useState('');
  const [selectedColor, setSelectedColor] = useState('#3b82f6');
  const [colorPickerOpen, setColorPickerOpen] = useState(false);

  const usedColors = objects.map(obj => obj.color);

  const addObject = () => {
    if (!newObjectName.trim()) return;

    // Ensure color is unique
    let finalColor = selectedColor;
    if (usedColors.includes(selectedColor)) {
      finalColor = generateUniqueColor(usedColors);
    }

    const newObject: AnnotationObject = {
      id: Date.now().toString(),
      name: newObjectName.trim(),
      color: finalColor,
      annotations: []
    };

    onObjectsChange([...objects, newObject]);
    setNewObjectName('');
    
    // Auto-select the new object
    onObjectSelect(newObject.id);
    
    // Generate new unique color for next object
    setSelectedColor(generateUniqueColor([...usedColors, finalColor]));
  };

  const removeObject = (objectId: string) => {
    const updatedObjects = objects.filter(obj => obj.id !== objectId);
    onObjectsChange(updatedObjects);
    
    if (selectedObject === objectId) {
      onObjectSelect(null);
    }

    // Notify parent component about object deletion for annotation cleanup
    onObjectDeleted?.(objectId);
  };

  const updateObjectColor = (objectId: string, color: string) => {
    // Check if color is already in use by another object
    const isColorInUse = objects.some(obj => obj.id !== objectId && obj.color === color);
    if (isColorInUse) {
      return; // Don't allow duplicate colors
    }

    const updatedObjects = objects.map(obj => 
      obj.id === objectId ? { ...obj, color } : obj
    );
    onObjectsChange(updatedObjects);
  };

  const handleColorChange = (color: string) => {
    if (!usedColors.includes(color)) {
      setSelectedColor(color);
    }
  };

  return (
    <div className="space-y-4">
      {/* Annotation Mode Selector */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-sm">Annotation Mode</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={() => onAnnotationModeChange('positive')}
              variant={annotationMode === 'positive' ? 'default' : 'outline'}
              className={`w-full text-xs ${
                annotationMode === 'positive' 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-slate-700 border-slate-600 text-white hover:bg-slate-600'
              }`}
            >
              <Plus className="h-3 w-3 mr-1" />
              Positive
            </Button>
            <Button
              onClick={() => onAnnotationModeChange('negative')}
              variant={annotationMode === 'negative' ? 'destructive' : 'outline'}
              className={`w-full text-xs ${
                annotationMode === 'negative' 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-slate-700 border-slate-600 text-white hover:bg-slate-600'
              }`}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Negative
            </Button>
          </div>
          <div className="text-xs text-slate-400">
            {annotationMode === 'positive' 
              ? 'Click to add positive points (include object)' 
              : 'Click to add negative points (exclude region)'
            }
          </div>
        </CardContent>
      </Card>

      {/* Object Manager */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-sm flex items-center gap-2">
            <Target className="h-4 w-4" />
            Objects ({objects.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Add New Object */}
          <div className="space-y-3">
            <div>
              <Label htmlFor="objectName" className="text-slate-300 text-xs">
                Object Name
              </Label>
              <Input
                id="objectName"
                value={newObjectName}
                onChange={(e) => setNewObjectName(e.target.value)}
                placeholder="Enter object name..."
                className="bg-slate-700 border-slate-600 text-white text-sm"
                onKeyPress={(e) => e.key === 'Enter' && addObject()}
              />
            </div>
            
            <div>
              <Label className="text-slate-300 text-xs mb-2 block">Color</Label>
              <Popover open={colorPickerOpen} onOpenChange={setColorPickerOpen}>
                <PopoverTrigger asChild>
                  <button
                    className="flex items-center gap-2 w-full p-2 bg-slate-700 border border-slate-600 rounded text-white text-sm hover:bg-slate-600"
                  >
                    <div
                      className="w-4 h-4 rounded border border-slate-500"
                      style={{ backgroundColor: selectedColor }}
                    />
                    <Palette className="h-3 w-3" />
                    Select Color
                  </button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-3" side="right">
                  <HexColorPicker 
                    color={selectedColor} 
                    onChange={handleColorChange}
                  />
                  <div className="mt-2 text-xs text-slate-600">
                    {usedColors.includes(selectedColor) && (
                      <p className="text-red-500">Color already in use</p>
                    )}
                  </div>
                </PopoverContent>
              </Popover>
            </div>

            <Button
              onClick={addObject}
              disabled={!newObjectName.trim() || usedColors.includes(selectedColor)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-sm disabled:opacity-50"
            >
              <Plus className="h-3 w-3 mr-1" />
              Add Object
            </Button>
          </div>

          {/* Objects List */}
          <div className="space-y-2">
            {objects.length === 0 ? (
              <div className="text-slate-400 text-xs text-center py-4">
                No objects created yet
              </div>
            ) : (
              objects.map((object) => (
                <div
                  key={object.id}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedObject === object.id
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-slate-600 bg-slate-700 hover:border-slate-500'
                  }`}
                  onClick={() => onObjectSelect(object.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Circle 
                        className="h-3 w-3 flex-shrink-0" 
                        style={{ color: object.color, fill: object.color }} 
                      />
                      <span className="text-white text-xs font-medium truncate">
                        {object.name}
                      </span>
                      {selectedObject === object.id && (
                        <Badge variant="secondary" className="text-xs">
                          Active
                        </Badge>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {/* Color picker for existing objects */}
                      <Popover>
                        <PopoverTrigger asChild>
                          <button
                            onClick={(e) => e.stopPropagation()}
                            className="w-5 h-5 rounded border border-slate-500 hover:border-white transition-colors"
                            style={{ backgroundColor: object.color }}
                          />
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-3" side="right">
                          <HexColorPicker 
                            color={object.color} 
                            onChange={(color) => updateObjectColor(object.id, color)}
                          />
                          <div className="mt-2 text-xs text-slate-600">
                            {usedColors.filter(c => c !== object.color).includes(object.color) && (
                              <p className="text-red-500">Color already in use</p>
                            )}
                          </div>
                        </PopoverContent>
                      </Popover>
                      
                      <Button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeObject(object.id);
                        }}
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-slate-400 hover:text-red-400"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="text-xs text-slate-400 mt-1">
                    {object.annotations.length} annotations
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
